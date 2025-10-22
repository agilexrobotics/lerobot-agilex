#!/home/agilex/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""

import sys
import os
# 添加LeRobot源码路径到Python路径
sys.path.insert(0, '../src')

import torch
import numpy as np

# Fix numpy compatibility issue with cv_bridge
import numpy
if not hasattr(numpy, '_ARRAY_API'):
    numpy._ARRAY_API = True
import pickle
import argparse
from einops import rearrange
from collections import deque
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header, String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Image, CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import threading
from torchvision import transforms
import cv2
# from aloha_msgs.srv import StatusSrv, StatusSrvRequest, StatusSrvResponse
from functools import partial
from tf_transformations import quaternion_from_euler, euler_from_quaternion
import sys
from scipy.spatial import geometric_slerp
import open3d as o3d
from sensor_msgs.msg import PointCloud2, PointField#, point_cloud2
from tf2_ros import Buffer, TransformListener
import math
import random
import time
from PIL import Image as PImage
import yaml

# 导入模型创建函数
import logging
from copy import copy
from pprint import pformat
from dataclasses import asdict
from torch import Tensor, nn
from contextlib import nullcontext
from multiprocessing import Process, shared_memory, Array
from multiprocessing import shared_memory, Array, Lock

from lerobot.policies.act.modeling_act import ACTPolicy as Policy
# from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy as Policy
from lerobot.utils.utils import get_safe_torch_device

try:
    from hbm_runtime import HB_HBMRuntime
    print("using: hbm_runtime")
except ImportError:
    print("hbm_runtime not found, please check!")
    exit()

inference_thread = None
inference_lock = threading.Lock()
inference_actions = None
inference_timestep = None
pre_instruction = ""
pre_instruction_attention_vector = None
pre_instruction_input_ids = None
pre_instruction_attention_mask = None
pre_batch = None


class RDK_ACTConfig:
    """Simple configuration class for RDK_ACTPolicy to match LeRobot policy interface."""
    def __init__(self, device="cpu", n_action_steps=50):
        self.device = device
        self.n_action_steps = n_action_steps
        self.use_amp = False  # Automatic Mixed Precision not used for BPU


# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
class RDK_ACTPolicy_Dynamic():
    def __init__(self, bpu_act_model_path, camera_names, n_action_steps=50):
        self.config = RDK_ACTConfig(device="cpu", n_action_steps=n_action_steps)
        self.n_action_steps = n_action_steps
        self._action_queue = deque([], maxlen=self.n_action_steps)
        self.camera_names = camera_names
        
        print(f"Initializing BPU policy with cameras: {camera_names}")
        
        # 动态加载归一化参数
        self.camera_params = {}
        for camera_name in camera_names:
            std_path = os.path.join(bpu_act_model_path, f"{camera_name}_std.npy")
            mean_path = os.path.join(bpu_act_model_path, f"{camera_name}_mean.npy")
            
            if os.path.exists(std_path) and os.path.exists(mean_path):
                self.camera_params[camera_name] = {
                    'std': torch.tensor(np.load(std_path), dtype=torch.float32) + 1e-8,
                    'mean': torch.tensor(np.load(mean_path), dtype=torch.float32)
                }
                print(f"Loaded normalization params for {camera_name}")
            else:
                raise FileNotFoundError(f"Missing normalization files for camera: {camera_name}")
        
        # 加载动作归一化参数
        action_std_path = os.path.join(bpu_act_model_path, "action_std.npy")
        action_mean_path = os.path.join(bpu_act_model_path, "action_mean.npy")
        action_std_unnormalize_path = os.path.join(bpu_act_model_path, "action_std_unnormalize.npy")
        action_mean_unnormalize_path = os.path.join(bpu_act_model_path, "action_mean_unnormalize.npy")
        
        # 检查所有必需文件
        required_files = [action_std_path, action_mean_path, action_std_unnormalize_path, action_mean_unnormalize_path]
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        self.action_std = torch.tensor(np.load(action_std_path), dtype=torch.float32) + 1e-8
        self.action_mean = torch.tensor(np.load(action_mean_path), dtype=torch.float32)
        self.action_std_unnormalize = torch.tensor(np.load(action_std_unnormalize_path), dtype=torch.float32)
        self.action_mean_unnormalize = torch.tensor(np.load(action_mean_unnormalize_path), dtype=torch.float32)

        # 验证参数
        for camera_name in camera_names:
            params = self.camera_params[camera_name]
            assert not torch.isinf(params['std']).any(), f"Invalid std for {camera_name}"
            assert not torch.isinf(params['mean']).any(), f"Invalid mean for {camera_name}"
            
        assert not torch.isinf(self.action_std).any(), "Invalid action_std"
        assert not torch.isinf(self.action_mean).any(), "Invalid action_mean"
        assert not torch.isinf(self.action_std_unnormalize).any(), "Invalid action_std_unnormalize"
        assert not torch.isinf(self.action_mean_unnormalize).any(), "Invalid action_mean_unnormalize"
        
        # 加载BPU模型
        bpu_act_policy_visionencoder_path = os.path.join(bpu_act_model_path,"BPU_ACTPolicy_VisionEncoder.hbm")
        bpu_act_policy_transformerlayers_path = os.path.join(bpu_act_model_path,"BPU_ACTPolicy_TransformerLayers.hbm")
        
        if not os.path.exists(bpu_act_policy_visionencoder_path):
            raise FileNotFoundError(f"Vision encoder model not found: {bpu_act_policy_visionencoder_path}")
        if not os.path.exists(bpu_act_policy_transformerlayers_path):
            raise FileNotFoundError(f"Transformer model not found: {bpu_act_policy_transformerlayers_path}")
            
        self.bpu_policy = HB_HBMRuntime([
            bpu_act_policy_visionencoder_path,
            bpu_act_policy_transformerlayers_path
        ])
        self.cnt = 0
        print("BPU models loaded successfully")
    
    def reset(self):
        """Reset the policy state, clear action queue."""
        self._action_queue.clear()
        
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        # normalize inputs
        batch = self.normalize_inputs(batch)

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            begin_time = time.time()
            
            # 准备状态输入
            state = batch["observation.state"].numpy().copy()
            #state = np.asarray(state, dtype=np.float32)
            #state = np.ascontiguousarray(state)
            #print(f"State input shape: {state.shape}")
            # 动态处理所有相机的视觉特征
            vision_features = []
            for camera_name in self.camera_names:
                camera_input = batch[f'observation.images.{camera_name}'].numpy().copy()
                #camera_input = np.asarray(camera_input, dtype=np.float32)
                #camera_input = np.ascontiguousarray(camera_input)
                # BPU模型期望NCHW格式 (1,3,720,1280)，不需要转置
                #print(f"Camera {camera_name} input shape:", camera_input.shape)
                # 通过VisionEncoder获取特征
                #print("BPU expects:", self.bpu_policy.input_shapes)
                #print("actually :", camera_input.shape, camera_input.dtype, camera_input.strides)
                vision_output = self.bpu_policy.run({"images": camera_input}, model_name="BPU_ACTPolicy_VisionEncoder")
                vision_feature = next(iter(vision_output["BPU_ACTPolicy_VisionEncoder"].values()))
                vision_features.append(vision_feature)
            
            # 构建TransformerLayers的输入
            transformer_inputs = {"states": state}
            for i, camera_name in enumerate(self.camera_names):
                transformer_inputs[f"{camera_name}_features"] = vision_features[i]
            
            # TransformerLayers推理
            transformer_outputs = self.bpu_policy.run(transformer_inputs, model_name="BPU_ACTPolicy_TransformerLayers")
            
            # 提取动作预测
            action_output = next(iter(transformer_outputs["BPU_ACTPolicy_TransformerLayers"].values()))
            actions = torch.from_numpy(action_output)[:, :self.n_action_steps]
            
            print(f"{self.cnt} BPU ACT Policy Time (dynamic {len(self.camera_names)} cameras): " + "\033[1;31m" + "%.2f ms"%(1000*(time.time() - begin_time)) + "\033[0m")
            self.cnt += 1
            
            actions = self.unnormalize_outputs({"action": actions})["action"]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()
    
    def normalize_inputs(self, batch):
        # 归一化状态
        batch["observation.state"] = (batch["observation.state"] - self.action_mean) / self.action_std
        
        # 动态归一化所有相机图像
        for camera_name in self.camera_names:
            if f'observation.images.{camera_name}' in batch:
                params = self.camera_params[camera_name]
                batch[f'observation.images.{camera_name}'] = (
                    batch[f'observation.images.{camera_name}'] - params['mean']
                ) / params['std']
        
        return batch
    
    def unnormalize_outputs(self, batch):
        batch["action"] = batch["action"] * self.action_std_unnormalize + self.action_mean_unnormalize
        return batch


class BlockingDeque:
    def __init__(self):
        self.deque = deque()
        self.not_empty = threading.Condition()

    def append(self, item):
        with self.not_empty:
            self.deque.append(item)
            self.not_empty.notify()

    def popleft(self):
        with self.not_empty:
            while len(self.deque) == 0:
                self.not_empty.wait()
            item = self.deque.popleft()
        return item

    def left(self):
        with self.not_empty:
            while len(self.deque) == 0:
                self.not_empty.wait()
            item = self.deque[0]
            return item

    def right(self):
        with self.not_empty:
            while len(self.deque) == 0:
                self.not_empty.wait()
            item = self.deque[-1]
            return item

    def size(self):
        with self.not_empty:
            return len(self.deque)


def matrix_to_xyzrpy(matrix):
    x = matrix[0, 3]
    y = matrix[1, 3]
    z = matrix[2, 3]
    roll = math.atan2(matrix[2, 1], matrix[2, 2])
    pitch = math.asin(-matrix[2, 0])
    yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    return [x, y, z, roll, pitch, yaw]


def create_transformation_matrix(x, y, z, roll, pitch, yaw):
    transformation_matrix = np.eye(4)
    A = np.cos(yaw)
    B = np.sin(yaw)
    C = np.cos(pitch)
    D = np.sin(pitch)
    E = np.cos(roll)
    F = np.sin(roll)
    DE = D * E
    DF = D * F
    transformation_matrix[0, 0] = A * C
    transformation_matrix[0, 1] = A * DF - B * E
    transformation_matrix[0, 2] = B * F + A * DE
    transformation_matrix[0, 3] = x
    transformation_matrix[1, 0] = B * C
    transformation_matrix[1, 1] = A * E + B * DF
    transformation_matrix[1, 2] = B * DE - A * F
    transformation_matrix[1, 3] = y
    transformation_matrix[2, 0] = -D
    transformation_matrix[2, 1] = C * F
    transformation_matrix[2, 2] = C * E
    transformation_matrix[2, 3] = z
    transformation_matrix[3, 0] = 0
    transformation_matrix[3, 1] = 0
    transformation_matrix[3, 2] = 0
    transformation_matrix[3, 3] = 1
    return transformation_matrix


def depth_to_color_projection(depth_image, color_intrinsics, depth_intrinsics, extrinsics):
    # 获取深度图像的宽度和高度
    depth_height, depth_width = depth_image.shape[:2]

    # 创建网格坐标
    u, v = np.meshgrid(np.arange(depth_width), np.arange(depth_height))
    u = u.flatten()
    v = v.flatten()
    depth_values = depth_image.flatten()

    # 将像素坐标转换为齐次坐标
    depth_points = np.vstack((u, v, np.ones_like(u)))

    # 将深度图像中的点转换到深度相机坐标系
    X_depth = np.linalg.inv(depth_intrinsics) @ depth_points

    # 将深度相机坐标系中的点转换到彩色相机坐标系
    X_color = extrinsics @ np.vstack((X_depth, np.ones((1, X_depth.shape[1]))))

    # 将彩色相机坐标系中的点投影到彩色图像平面
    x_color = (color_intrinsics[0, 0] * (X_color[0, :] / X_color[2, :]) + color_intrinsics[0, 2]).round().astype(int)
    y_color = (color_intrinsics[1, 1] * (X_color[1, :] / X_color[2, :]) + color_intrinsics[1, 2]).round().astype(int)

    # 创建对齐后的深度图像
    aligned_depth = np.zeros_like(depth_image)

    # 将投影后的点存储到对齐后的深度图像中
    valid_indices = (x_color >= 0) & (x_color < depth_image.shape[1]) & (y_color >= 0) & (y_color < depth_image.shape[0])
    aligned_depth[y_color[valid_indices], x_color[valid_indices]] = depth_values[valid_indices]

    return aligned_depth


def color_depth_to_point_cloud(color_image, depth_image, color_intrinsic, depth_intrinsic, color_extrinsic, depth_extrinsic):
    if not np.array_equal(color_extrinsic, depth_extrinsic):
        depth_image = depth_to_color_projection(depth_image, color_intrinsic, depth_intrinsic, np.dot(np.linalg.inv(color_extrinsic), depth_extrinsic))
        # 相机内参矩阵
        fx, fy = color_intrinsic[0][0], color_intrinsic[1][1]
        cx, cy = color_intrinsic[0][2], color_intrinsic[1][2]
    else:
        # 相机内参矩阵
        fx, fy = depth_intrinsic[0][0], depth_intrinsic[1][1]
        cx, cy = depth_intrinsic[0][2], depth_intrinsic[1][2]
    # 获取图像的宽度和高度
    height, width = depth_image.shape

    u, v = np.meshgrid(np.arange(width), np.arange(height))
    u = u.astype(np.float32)
    v = v.astype(np.float32)
    z = depth_image.astype(np.float32) / 1000.0  # 将深度图像转换为米

    # 计算 3D 坐标
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # 提取 color 颜色值
    b = color_image[..., 0].astype(np.float32)
    g = color_image[..., 1].astype(np.float32)
    r = color_image[..., 2].astype(np.float32)

    # 合并为点云
    point_cloud = np.stack((x, y, z, r, g, b), axis=-1)

    # 跳过深度为零的点
    valid_mask = z > 0.0
    point_cloud = point_cloud[valid_mask]

    return point_cloud


def get_camera_color(camera_names, history_num, camera_color):
    colorss = []
    for cam_name in camera_names:
        colors = []
        for i in range(history_num):
            color = camera_color[cam_name][i]
            color = cv2.imencode('.jpg', color)[1].tobytes()
            color = cv2.imdecode(np.frombuffer(color, np.uint8), cv2.IMREAD_COLOR)
            default_width = 1280
            default_height = 720
            camera_width = color.shape[1]
            camera_height = color.shape[0]
            width_diff = default_width - camera_width
            height_diff = default_height - camera_height
            if width_diff < 0:
                clip_width = abs(width_diff) // 2
                color = color[:, clip_width:clip_width + default_width]
            elif width_diff > 0:
                add_width = width_diff // 2
                top, bottom, left, right = 0, 0, add_width, add_width
                color = cv2.copyMakeBorder(color, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
            if height_diff < 0:
                clip_height = abs(height_diff) // 2
                color = color[clip_height:clip_height + default_height, :]
            elif height_diff > 0:
                add_height = height_diff // 2
                top, bottom, left, right = add_height, add_height, 0, 0
                color = cv2.copyMakeBorder(color, top, bottom, left, right, cv2.BORDER_CONSTANT,value=0)
            color = rearrange(color, 'h w c -> c h w')
            colors.append(color)
        colorss.append(np.stack(colors, axis=0))
    colorss = np.stack(colorss, axis=0)
    colorss = colorss.astype(np.float32)[np.newaxis, :]
    return colorss  # b, num_camera, num_history, c, h, w


def get_camera_depth(camera_names, history_num, camera_depth):
    depthss = []
    for cam_name in camera_names:
        depths = []
        for i in range(history_num):
            depth = camera_depth[cam_name][i]
            default_width = 1280
            default_height = 720
            camera_width = depth.shape[1]
            camera_height = depth.shape[0]
            width_diff = default_width - camera_width
            height_diff = default_height - camera_height
            if width_diff < 0:
                clip_width = abs(width_diff) // 2
                depth = depth[:, clip_width:clip_width + default_width]
            elif width_diff > 0:
                add_width = width_diff // 2
                top, bottom, left, right = 0, 0, add_width, add_width
                depth = cv2.copyMakeBorder(depth, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
            if height_diff < 0:
                clip_height = abs(height_diff) // 2
                depth = depth[clip_height:clip_height + default_height, :]
            elif height_diff > 0:
                add_height = height_diff // 2
                top, bottom, left, right = add_height, add_height, 0, 0
                depth = cv2.copyMakeBorder(depth, top, bottom, left, right, cv2.BORDER_CONSTANT,value=0)
            depths.append(depth)
        depthss.append(np.stack(depths, axis=0))
    depthss = np.stack(depthss, axis=0)
    depthss = depthss.astype(np.float32)[np.newaxis, :]
    return depthss  # b, num_camera, num_history, h, w


def get_camera_point_cloud(camera_names, history_num, voxel_size, use_farthest_point_down_sample,
                           point_num, use_camera_point_cloud_rgb,
                           camera_point_cloud, color_extrinsics, point_cloud_extrinsics):
    camera_point_cloudss = []
    for cam_id, cam_name in enumerate(camera_names):
        camera_point_clouds = []
        for i in range(history_num):
            pc = camera_point_cloud[cam_name][i]

            if voxel_size != 0:
                condition = pc[:, 2] < 2
                pc = pc[condition, :]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pc[:, :3].astype(np.float64))
                if pc.shape[1] != 6:
                    rgbs = pc[:, 3].view(np.uint32)
                    # rgbs = pc[:, 3].view(np.uint64)
                    r = (np.right_shift(rgbs, 16) % 256)[:, np.newaxis]
                    g = (np.right_shift(rgbs, 8) % 256)[:, np.newaxis]
                    b = (rgbs % 256)[:, np.newaxis]
                    r_g_b = np.concatenate([r, g, b], axis=-1)
                    pcd.colors = o3d.utility.Vector3dVector(r_g_b.astype(np.float64))
                else:
                    pcd.colors = o3d.utility.Vector3dVector(pc[:, 3:].astype(np.float64))

                downsampled_cloud = pcd.voxel_down_sample(voxel_size)
                if use_farthest_point_down_sample and len(downsampled_cloud.points) > point_num:
                    downsampled_cloud = downsampled_cloud.farthest_point_down_sample(point_num)

                pc = np.concatenate([downsampled_cloud.points, downsampled_cloud.colors], axis=-1)
                if pc.shape[0] > point_num:
                    idxs = np.random.choice(pc.shape[0], point_num, replace=False)
                    pc = pc[idxs]
                elif pc.shape[0] < point_num:
                    if pc.shape[0] == 0:
                        pc = np.zeros([1, 4], dtype=np.float32)
                    idxs1 = np.arange(pc.shape[0])
                    idxs2 = np.random.choice(pc.shape[0], point_num - pc.shape[0], replace=True)
                    idxs = np.concatenate([idxs1, idxs2], axis=0)
                    pc = pc[idxs]
            else:
                condition = pc[:, 2] < 2
                pc = pc[condition, :]
                if pc.shape[0] >= point_num:
                    idxs = np.random.choice(pc.shape[0], point_num, replace=False)
                elif pc.shape[0] < point_num:
                    idxs1 = np.arange(pc.shape[0])
                    idxs2 = np.random.choice(pc.shape[0], point_num - pc.shape[0], replace=True)
                    idxs = np.concatenate([idxs1, idxs2], axis=0)
                if pc.shape[1] != 6:
                    # rgbs = pc[idxs][:, 3].view(np.uint64)
                    rgbs = pc[idxs][:, 3].view(np.uint32)
                    r = (np.right_shift(rgbs, 16) % 256)[:, np.newaxis]
                    g = (np.right_shift(rgbs, 8) % 256)[:, np.newaxis]
                    b = (rgbs % 256)[:, np.newaxis]
                    r_g_b = np.concatenate([r, g, b], axis=-1)
                    pc = np.concatenate([pc[idxs][:, :3], r_g_b], axis=-1)
                else:
                    pc = pc[idxs]

            if not np.array_equal(color_extrinsics[cam_id], point_cloud_extrinsics[cam_id]):
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
                pc[:, 3:] = pc[:, 3:] / 255
                pcd.colors = o3d.utility.Vector3dVector(pc[:, 3:])
                pcd.transform(np.dot(np.linalg.inv(color_extrinsics[cam_id]), point_cloud_extrinsics[cam_id]))
                pcd.colors = o3d.utility.Vector3dVector(
                    (np.asarray(pcd.colors) * 255).astype(np.float64))
                pc = np.concatenate([pcd.points, pcd.colors], axis=-1)

            index = 6 if use_camera_point_cloud_rgb else 3
            camera_point_clouds.append(pc[:, :index])
        camera_point_cloudss.append(np.stack(camera_point_clouds, axis=0))
    camera_point_cloudss = camera_point_cloudss.astype(np.float32)[np.newaxis, :]
    return camera_point_cloudss  # b, num_camera, num_history, point_num, 6 or 3


def point_cloud_to_numpy(point_cloud):
    dtype_list = []
    for field in point_cloud.fields:
        if field.datatype == PointField.FLOAT32:
            dtype_list.append((field.name, np.float32))
        elif field.datatype == PointField.UINT32:
            dtype_list.append((field.name, np.uint32))
        elif field.datatype == PointField.UINT8:
            dtype_list.append((field.name, np.uint8))
    # Create a numpy array using the built dtype_list
    cloud_arr = np.frombuffer(point_cloud.data, dtype=np.dtype(dtype_list))
    # Extract x, y, z and rgb values
    x = cloud_arr['x']
    y = cloud_arr['y']
    z = cloud_arr['z']
    rgb = cloud_arr['rgb']#.view(np.uint32)
    pc_array = np.vstack([x, y, z, rgb]).T
    return pc_array


def make_policy(args):
    """
    Create LeRobot ACT policy from pretrained model
    """
    # Configure device
    device = get_safe_torch_device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load custom stats from file if provided
    dataset_stats = None
    if hasattr(args, 'stats_path') and args.stats_path:
        print(f"Loading custom stats from file: {args.stats_path}")
        import json
        try:
            with open(args.stats_path, 'r') as f:
                stats_data = json.load(f)
            
            # 转换为LeRobot期望的格式
            dataset_stats = {}
            for key, stats in stats_data.items():
                dataset_stats[key] = {}
                for stat_name, values in stats.items():
                    dataset_stats[key][stat_name] = torch.tensor(values, dtype=torch.float32)
            
            print(f"Loaded stats from file for keys: {list(dataset_stats.keys())}")
                    
        except Exception as e:
            print(f"Failed to load stats from file: {e}")
            dataset_stats = None
    
    
    # Load the policy using LeRobot's standard interface
    if dataset_stats is not None:
        try:
            # 先加载默认policy
            # policy = Policy.from_pretrained(args.pretrained_model_name_or_path)
            policy = RDK_ACTPolicy_Dynamic(args.pretrained_model_name_or_path, args.camera_color_names)
            # 然后手动设置normalization的stats
            if hasattr(policy, 'normalize_inputs') and hasattr(policy, 'unnormalize_outputs'):
                # 更新normalize_inputs的stats
                for key, stats in dataset_stats.items():
                    if key.startswith('observation'):
                        buffer_name = f'buffer_{key.replace(".", "_")}'
                        if hasattr(policy.normalize_inputs, buffer_name):
                            buffer = getattr(policy.normalize_inputs, buffer_name)
                            for stat_name, stat_value in stats.items():
                                if stat_name in buffer:
                                    # 确保stat_value是torch tensor
                                    if not isinstance(stat_value, torch.Tensor):
                                        stat_value = torch.tensor(stat_value, dtype=torch.float32)
                                    else:
                                        stat_value = stat_value.float()
                                    
                                    # 确保维度匹配
                                    if buffer[stat_name].shape != stat_value.shape:
                                        # 尝试reshape或者跳过
                                        if buffer[stat_name].numel() == stat_value.numel():
                                            stat_value = stat_value.reshape(buffer[stat_name].shape)
                                        else:
                                            continue
                                    
                                    buffer[stat_name].copy_(stat_value)
                    elif key == 'action':
                        # 更新unnormalize_outputs的stats
                        if hasattr(policy.unnormalize_outputs, 'buffer_action'):
                            buffer = getattr(policy.unnormalize_outputs, 'buffer_action')
                            for stat_name, stat_value in stats.items():
                                if stat_name in buffer:
                                    # 确保stat_value是torch tensor
                                    if not isinstance(stat_value, torch.Tensor):
                                        stat_value = torch.tensor(stat_value, dtype=torch.float32)
                                    else:
                                        stat_value = stat_value.float()
                                    
                                    # 确保维度匹配
                                    if buffer[stat_name].shape != stat_value.shape:
                                        # 尝试reshape或者跳过
                                        if buffer[stat_name].numel() == stat_value.numel():
                                            stat_value = stat_value.reshape(buffer[stat_name].shape)
                                        else:
                                            continue
                                    
                                    buffer[stat_name].copy_(stat_value)
            
            print("Policy loaded with custom stats")
        except Exception as e:
            print(f"Failed to load policy with custom stats: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to default loading
            # policy = Policy.from_pretrained(args.pretrained_model_name_or_path)
            policy = RDK_ACTPolicy_Dynamic(args.pretrained_model_name_or_path, args.camera_color_names)
            print("Policy loaded with default stats (fallback)")
    else:
        # 使用默认方式加载
        # policy = Policy.from_pretrained(args.pretrained_model_name_or_path)
        policy = RDK_ACTPolicy_Dynamic(args.pretrained_model_name_or_path, args.camera_color_names)
        print("Policy loaded with default stats")
    
    # Move policy to device
    #policy.to(device)
    
    # Set to evaluation mode
    #policy.eval()
    
    print(f"Policy loaded on device: {device}")
    
    # 检查是否有无限值（表示stats没有正确加载）
    if hasattr(policy, 'normalize_inputs'):
        for key in ['observation.state']:
            buffer_name = f'buffer_{key.replace(".", "_")}'
            if hasattr(policy.normalize_inputs, buffer_name):
                buffer = getattr(policy.normalize_inputs, buffer_name)
                if 'mean' in buffer:
                    mean_val = buffer['mean']
                    std_val = buffer['std']
                    if torch.isinf(mean_val).any():
                        print(f"WARNING: {key} mean contains inf values!")
                    if torch.isinf(std_val).any():
                        print(f"WARNING: {key} std contains inf values!")
    
    if hasattr(policy, 'unnormalize_outputs'):
        if hasattr(policy.unnormalize_outputs, 'buffer_action'):
            buffer = getattr(policy.unnormalize_outputs, 'buffer_action')
            if 'mean' in buffer:
                mean_val = buffer['mean']
                std_val = buffer['std']
                if torch.isinf(mean_val).any():
                    print(f"WARNING: Action mean contains inf values!")
                if torch.isinf(std_val).any():
                    print(f"WARNING: Action std contains inf values!")
    
    return policy


def inference_process(args, t, ros_operator, policy, infer):
    global inference_lock
    global inference_actions
    global inference_timestep
    global pre_instruction
    global pre_instruction_attention_vector
    global pre_instruction_input_ids
    global pre_instruction_attention_mask
    global pre_batch
    bridge = CvBridge()
    if not infer and pre_batch is not None:
        start_time = time.time()
        
        # 使用LeRobot标准推理接口
        with torch.no_grad():
            action = policy.select_action(pre_batch)
        
        inference_time = time.time() - start_time
        
        inference_lock.acquire()
        # 正确处理action维度
        if action.dim() == 1:  # [action_dim]
            inference_actions = action.cpu().numpy()[np.newaxis, np.newaxis, :]  # [1, 1, action_dim]
        elif action.dim() == 2:  # [1, action_dim] 
            inference_actions = action.cpu().numpy()[:, np.newaxis, :]  # [1, 1, action_dim]
        else:  # [1, chunk_size, action_dim]
            inference_actions = action.cpu().numpy()
        
        inference_timestep = t
        inference_lock.release()
        return
    (instruction,
     camera_colors, camera_depths, camera_point_clouds,
     arm_joint_states, arm_end_poses,  robot_base_vels, last_ctrl_arm_end_poses) = ros_operator.get_frame()

    camera_color_dict = {}
    camera_color = None
    camera_depth_dict = {}
    camera_depth = None
    camera_point_cloud_dict = {}
    camera_point_cloud = None
    for j in range(len(args.camera_color_names)):
        camera_color_dict[args.camera_color_names[j]] = [bridge.imgmsg_to_cv2(camera_colors[i][j], 'rgb8') for i in range(args.obs_history_num)]
    if args.use_camera_color:
        camera_color = get_camera_color(args.camera_color_names, args.obs_history_num, camera_color_dict)  # b, num_camera, history, c, h, w

    if args.use_camera_color_depth_to_point_cloud or args.use_camera_depth:
        for j in range(len(args.camera_depth_names)):
            if np.array_equal(ros_operator.get_camera_color_extrinsic(j), ros_operator.get_camera_depth_extrinsic(j)):
                camera_depth_dict[args.camera_depth_names[j]] = [bridge.imgmsg_to_cv2(camera_depths[i][j], 'passthrough') for i in range(args.obs_history_num)]
            else:
                camera_depth_dict[args.camera_depth_names[j]] = [depth_to_color_projection(bridge.imgmsg_to_cv2(camera_depths[i][j], 'passthrough'),
                                                                                            ros_operator.get_camera_color_intrinsic(j), ros_operator.get_camera_depth_intrinsic(j),
                                                                                            np.dot(np.linalg.inv(ros_operator.get_camera_color_extrinsic(j)), ros_operator.get_camera_depth_extrinsic(j))) for i in range(args.obs_history_num)]
        if args.use_camera_depth:
            camera_depth = get_camera_depth(args.camera_depth_names, args.obs_history_num, camera_depth_dict)  # b, num_camera, num_history, h, w

        color_extrinsics = []
        point_cloud_extrinsics = []
        if args.use_camera_point_cloud:
            if not args.use_camera_color_depth_to_point_cloud:
                for j in range(len(args.camera_point_cloud_names)):
                    camera_point_cloud_dict[args.camera_point_cloud_names[j]] = [point_cloud_to_numpy(camera_point_clouds[i][j]) for i in range(args.obs_history_num)]
                    color_extrinsics.append(ros_operator.get_camera_color_extrinsic(j))
                    point_cloud_extrinsics.append(ros_operator.get_camera_point_cloud_extrinsic(j))
            else:
                for j in range(len(args.camera_point_cloud_names)):
                    color_intrinsic = ros_operator.get_camera_color_intrinsic(j)
                    depth_intrinsic = ros_operator.get_camera_color_intrinsic(j)
                    color_extrinsic = ros_operator.get_camera_color_extrinsic(j)
                    depth_extrinsic = ros_operator.get_camera_color_extrinsic(j)
                    color_extrinsics.append(color_extrinsic)
                    point_cloud_extrinsics.append(depth_extrinsic)
                    camera_point_cloud_dict[args.camera_point_cloud_names[j]] = [color_depth_to_point_cloud(camera_color_dict[args.camera_color_names[j]][i], camera_depth_dict[args.camera_depth_names[j]][i],
                                                                                                            color_intrinsic, depth_intrinsic,
                                                                                                            color_extrinsic, depth_extrinsic) for i in range(args.obs_history_num)]
            camera_point_cloud = get_camera_point_cloud(args.camera_point_cloud_names, args.obs_history_num,
                                                        args.camera_point_cloud_voxel_size, args.use_farthest_point_down_sample,
                                                        args.camera_point_cloud_point_num,
                                                        args.use_camera_point_cloud_rgb, camera_point_cloud_dict, 
                                                        color_extrinsics, point_cloud_extrinsics)  # b, num_camera, num_history, point_num, 6 or 3

    robot_state = []
    if args.use_arm_joint_state % 2 == 1:
        qpos_joint_state = [np.concatenate([np.array(arm_joint_states[i][j].position) for j in range(len(args.arm_joint_state_names))], axis=0)[np.newaxis, :] for i in range(args.obs_history_num)]
        qpos_joint_state = np.concatenate(qpos_joint_state, axis=0)  # obs_history_num, num_arm * 7
        robot_state.append(qpos_joint_state)
    if args.use_arm_end_pose % 2 == 1:
        qpos_end_pose = [np.concatenate(
            [np.array([arm_end_poses[i][j].pose.position.x, arm_end_poses[i][j].pose.position.y, arm_end_poses[i][j].pose.position.z,
                        arm_end_poses[i][j].pose.orientation.x, arm_end_poses[i][j].pose.orientation.y, arm_end_poses[i][j].pose.orientation.z, arm_end_poses[i][j].pose.orientation.w])
                for j in range(len(args.arm_end_pose_names))],
            axis=0)[np.newaxis, :] for i in range(args.obs_history_num)]
        qpos_end_pose = np.concatenate(qpos_end_pose, axis=0)  # obs_history_num, num_arm * 7
        robot_state.append(qpos_end_pose)
    if args.use_robot_base % 2 == 1:
        qpos_robot_base = [np.array([robot_base_vels[i][0].twist.twist.linear.x,
                                        robot_base_vels[i][0].twist.twist.linear.y,
                                        robot_base_vels[i][0].twist.twist.angular.z])[np.newaxis, :]
                            for i in range(args.obs_history_num)]
        qpos_robot_base = np.concatenate(qpos_robot_base, axis=0)  # obs_history_num, 3
        robot_state.append(qpos_robot_base)
    robot_state = np.concatenate(robot_state, axis=-1)[np.newaxis, :]  # b, obs_history_num, n
    
    # 准备batch数据用于LeRobot ACT推理
    #device = next(policy.parameters()).device
    
    # 准备observation state (最新的关节状态)
    observation_state = robot_state[0, -1, :]  # 取最新的状态
    observation_state = torch.from_numpy(observation_state).float().unsqueeze(0)#.to(device)  # [1, state_dim]
    
    # 准备图像数据 - 转换为torch tensor
    if camera_color is not None and args.use_camera_color:
        # camera_color shape: [batch, num_cameras, history, c, h, w]
        # LeRobot expects images as tensors in batch format
        observation_images = {}
        
        # 确保正确的索引
        if camera_color.ndim == 5:  # [num_cameras, history, c, h, w]
            for cam_idx, cam_name in enumerate(args.camera_color_names):
                # 取最新的图像帧
                img_tensor = torch.from_numpy(camera_color[cam_idx, -1]).float() / 255.0  # [c, h, w]
                img_tensor = img_tensor.unsqueeze(0)#.to(device)  # [1, c, h, w]
                observation_images[f"observation.images.{cam_name}"] = img_tensor
        elif camera_color.ndim == 6:  # [batch, num_cameras, history, c, h, w]
            for cam_idx, cam_name in enumerate(args.camera_color_names):
                # 取最新的图像帧
                img_tensor = torch.from_numpy(camera_color[0, cam_idx, -1]).float() / 255.0  # [c, h, w]
                img_tensor = img_tensor.unsqueeze(0)#.to(device)  # [1, c, h, w]
                observation_images[f"observation.images.{cam_name}"] = img_tensor
        else:
            print(f"Unexpected camera_color shape: {camera_color.shape}")
            return
    
    # 构造批次数据
    batch = {
        "observation.state": observation_state,
    }
    
    # 添加图像观测
    if camera_color is not None and args.use_camera_color:
        batch.update(observation_images)
    pre_batch = batch
    start_time = time.time()
    
    # 使用LeRobot标准推理接口
    with torch.no_grad():
        action = policy.select_action(batch)
    
    inference_time = time.time() - start_time
    
    inference_lock.acquire()
    # 正确处理action维度
    if action.dim() == 1:  # [action_dim]
        inference_actions = action.cpu().numpy()[np.newaxis, np.newaxis, :]  # [1, 1, action_dim]
    elif action.dim() == 2:  # [1, action_dim] 
        inference_actions = action.cpu().numpy()[:, np.newaxis, :]  # [1, 1, action_dim]
    else:  # [1, chunk_size, action_dim]
        inference_actions = action.cpu().numpy()
    
    inference_timestep = t
    inference_lock.release()


class RosOperator(Node):
    def __init__(self, args):
        super().__init__('inference')
        self.args = args
        self.bridge = CvBridge()

        # 创建共享的TF buffer和listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.instruction = args.instruction
        self.camera_color_deques = [BlockingDeque() for _ in range(len(args.camera_color_names))]
        self.camera_depth_deques = [BlockingDeque() for _ in range(len(args.camera_depth_names))]
        self.camera_point_cloud_deques = [BlockingDeque() for _ in range(len(args.camera_point_cloud_names))]
        self.arm_joint_state_deques = [BlockingDeque() for _ in range(len(args.arm_joint_state_names))]
        self.arm_end_pose_deques = [BlockingDeque() for _ in range(len(args.arm_end_pose_names))]
        self.robot_base_vel_deques = [BlockingDeque() for _ in range(len(args.robot_base_vel_names))]

        self.camera_color_intrinsics = [None for _ in range(len(self.args.camera_color_names))]
        self.camera_depth_intrinsics = [None for _ in range(len(self.args.camera_depth_names))]
        self.camera_point_cloud_intrinsics = [None for _ in range(len(self.args.camera_point_cloud_names))]
        self.camera_color_extrinsics = [None for _ in range(len(self.args.camera_color_names))]
        self.camera_depth_extrinsics = [None for _ in range(len(self.args.camera_depth_names))]
        self.camera_point_cloud_extrinsics = [None for _ in range(len(self.args.camera_point_cloud_names))]

        self.all_config_exist = False

        self.camera_color_history_list = []
        self.camera_depth_history_list = []
        self.camera_point_cloud_history_list = []
        self.arm_joint_state_history_list = []
        self.arm_end_pose_history_list = []
        self.robot_base_vel_history_list = []

        self.instruction_subscriber = self.create_subscription(String, self.args.instruction_topic, self.instruction_callback, 1)
        self.camera_color_subscriber = [self.create_subscription(Image, self.args.camera_color_topics[i], partial(self.camera_color_callback, i), 1) for i in range(len(self.args.camera_color_names))]
        self.camera_depth_subscriber = [self.create_subscription(Image, self.args.camera_depth_topics[i], partial(self.camera_depth_callback, i), 1) for i in range(len(self.args.camera_depth_names))]
        self.camera_point_cloud_subscriber = [self.create_subscription(PointCloud2, self.args.camera_point_cloud_topics[i], partial(self.camera_point_cloud_callback, i), 1) for i in range(len(self.args.camera_point_cloud_names))]
        self.arm_joint_state_subscriber = [self.create_subscription(JointState, self.args.arm_joint_state_topics[i], partial(self.arm_joint_state_callback, i), 1) for i in range(len(self.args.arm_joint_state_names))]
        self.arm_end_pose_subscriber = [self.create_subscription(PoseStamped, self.args.arm_end_pose_topics[i], partial(self.arm_end_pose_callback, i), 1) for i in range(len(self.args.arm_end_pose_names))]
        self.robot_base_vel_subscriber = [self.create_subscription(Odometry, self.args.robot_base_vel_topics[i], partial(self.robot_base_vel_callback, i), 1) for i in range(len(self.args.robot_base_vel_names))]

        self.camera_color_config_subscriber = [self.create_subscription(CameraInfo, self.args.camera_color_config_topics[i], partial(self.camera_color_config_callback, i), 1) for i in range(len(self.args.camera_color_names))]
        self.camera_depth_config_subscriber = [self.create_subscription(CameraInfo, self.args.camera_depth_config_topics[i], partial(self.camera_depth_config_callback, i), 1) for i in range(len(self.args.camera_depth_names))]
        self.camera_point_cloud_config_subscriber = [self.create_subscription(CameraInfo, self.args.camera_point_cloud_config_topics[i], partial(self.camera_point_cloud_config_callback, i), 1) for i in range(len(self.args.camera_point_cloud_names))]

        self.arm_joint_state_ctrl_publisher = [self.create_publisher(JointState, self.args.arm_joint_state_ctrl_topics[i], 10) for i in range(len(self.args.arm_joint_state_ctrl_topics))]
        self.arm_end_pose_ctrl_publisher = [self.create_publisher(PoseStamped, self.args.arm_end_pose_ctrl_topics[i], 10) for i in range(len(self.args.arm_end_pose_ctrl_topics))]
        self.robot_base_vel_ctrl_publisher = self.create_publisher(Twist, self.args.robot_base_vel_ctrl_topic, 10)

        # self.inference_status_service = self.create_service(StatusSrv, self.args.aloha_inference_status_service, self.change_inference_status)
        self.inference_status = -1

        self.arm_joint_state_ctrl_thread = None
        self.arm_joint_state_ctrl_thread_return_lock = threading.Lock()
        self.arm_joint_state_ctrl_thread_return_lock.acquire()

        self.last_ctrl_arm_joint_state = None

        self.k = 3
        self.times = np.array([i for i in range(self.k)])
        self.arm_joint_state_ctrl_history_list = []

        self.arm_end_pose_ctrl_thread = None
        self.arm_end_pose_ctrl_thread_return_lock = threading.Lock()
        self.arm_end_pose_ctrl_thread_return_lock.acquire()

        self.last_ctrl_arm_end_poses = []
        self.arm_end_pose_ctrl_history_list = []

        self.model_inference_thread = threading.Thread(target=self.model_inference)
        self.model_inference_thread.start()

    def model_inference(self):
        global inference_lock
        global inference_actions
        global inference_timestep
        global inference_thread
        global pre_instruction_attention_vector
        # instruction = np.load(self.args.lang_embeddings_path, allow_pickle=True).item()
        # instructions_vectors = instruction[f'full_instruction/vector']
        # instruction_index = random.randint(0, len(instructions_vectors) - 1)
        # pre_instruction_attention_vector = instructions_vectors[instruction_index]
        # pre_instruction_attention_vector = pre_instruction_attention_vector[np.newaxis, :]

        # 创建模型并加载语言嵌入
        policy = make_policy(self.args)

        joint_state0 = [[0, 0, 0, 0, 0, 0, 0.08] for _ in range(len(self.args.arm_joint_state_ctrl_topics))]
        joint_state1 = [[0, 0, 0, 0, 0, 0, 0.0] for _ in range(len(self.args.arm_joint_state_ctrl_topics))]
        end_pose0 = [[0.05, 0, 0.2, 0, 0, 0, 1.6] for _ in range(len(self.args.arm_end_pose_ctrl_topics))]
        end_pose1 = [[0.05, 0, 0.2, 0, 0, 0, 0.0] for _ in range(len(self.args.arm_end_pose_ctrl_topics))]
        robot_base_vel0 = [0, 0, 0]
        robot_base_vel1 = [0, 0, 0]
        if self.args.use_arm_joint_state > 1:
            self.arm_joint_state_ctrl_linear_interpolation_thread(joint_state0, True, calc_step=True)
        if self.args.use_arm_end_pose > 1:
            self.arm_end_pose_ctrl(end_pose0)
            self.arm_joint_state_ctrl(joint_state0)
            # self.arm_end_pose_ctrl_linear_interpolation_thread(end_pose0, True, calc_step=True)
        if self.args.use_robot_base > 1:
            self.robot_base_vel_ctrl(robot_base_vel0)
        pre_inference_status = -1
        ctrl_rate = self.create_rate(10)
        all_actions = None

        with torch.inference_mode():
            while rclpy.ok():
                t = 0
                max_t = 0
                if inference_thread is not None:
                    inference_thread.join()
                    inference_thread = None
                while t < self.args.max_publish_step and rclpy.ok():
                    inference_status = self.get_inference_status()
                    if inference_status == -1:
                        input("Please press any key to start inference:")
                        self.set_inference_status(0)
                    is_new_action = False
                    if self.args.asynchronous_inference:
                        if inference_thread is None and (self.args.pos_lookahead_step == 0 or t % self.args.pos_lookahead_step == 0 or t >= max_t) and self.check_frame():
                            inference_thread = threading.Thread(target=inference_process,
                                                                args=(self.args, t, self, policy, True))
                            inference_thread.start()
                            inference_thread.join()
                            inference_lock.acquire()
                            inference_thread = None
                            all_actions = inference_actions
                            t_start = inference_timestep
                            max_t = t_start + self.args.chunk_size
                            is_new_action = True
                            inference_lock.release()
                            print("infer")
                        else:
                            inference_thread = threading.Thread(target=inference_process,
                                                                args=(self.args, t, self, policy, False))
                            inference_thread.start()
                            inference_thread.join()
                            inference_lock.acquire()
                            inference_thread = None
                            all_actions = inference_actions
                            t_start = inference_timestep
                            inference_lock.release()
                            print("no infer")
                    if t >= max_t:
                        print("inference time error")
                        continue

                    raw_action = all_actions[:, t - t_start]
                    raw_action = raw_action[0]
                    if self.args.use_arm_joint_state > 1:
                        action_joint_state = raw_action[:self.args.arm_joint_state_dim * len(self.args.arm_joint_state_names)]
                        raw_action = raw_action[self.args.arm_joint_state_dim * len(self.args.arm_joint_state_names):]
                        action_joint_state = [action_joint_state[i*self.args.arm_joint_state_dim:(i+1)*self.args.arm_joint_state_dim] for i in range(len(self.args.arm_joint_state_names))]
                        self.arm_joint_state_ctrl_interpolation_thread(action_joint_state, self.args.blocking_publish, is_new_action if self.args.asynchronous_inference else False)
                        # self.arm_joint_state_ctrl(action_joint_state)
                        # ctrl_rate.sleep()
                    if self.args.use_arm_end_pose > 1:
                        action_end_pose = raw_action[:self.args.arm_end_pose_dim * len(self.args.arm_end_pose_names)]
                        raw_action = raw_action[self.args.arm_end_pose_dim * len(self.args.arm_end_pose_names):]
                        action_end_pose = [action_end_pose[i*self.args.arm_end_pose_dim:(i+1)*self.args.arm_end_pose_dim] for i in range(len(self.args.arm_end_pose_names))]
                        self.arm_end_pose_ctrl_linear_interpolation_thread(action_end_pose, self.args.blocking_publish, calc_step=False)
                        # self.arm_end_pose_ctrl(action_end_pose)
                        # ctrl_rate.sleep()
                    if self.args.use_robot_base > 1:
                        action_robot_base = raw_action[:self.args.robot_base_dim]
                        self.robot_base_vel_ctrl(action_robot_base)
                    print("t:", t, max_t)
                    t += 1

    def get_camera_color_intrinsic(self, index):
        return self.camera_color_intrinsics[index]

    def get_camera_depth_intrinsic(self, index):
        return self.camera_depth_intrinsics[index]

    def get_camera_point_cloud_intrinsic(self, index):
        return self.camera_point_cloud_intrinsics[index]

    def get_camera_color_extrinsic(self, index):
        return self.camera_color_extrinsics[index]

    def get_camera_depth_extrinsic(self, index):
        return self.camera_depth_extrinsics[index]

    def get_camera_point_cloud_extrinsic(self, index):
        return self.camera_point_cloud_extrinsics[index]

    def instruction_callback(self, msg):
        self.instruction = msg.data

    def camera_color_callback(self, index, msg):
        if self.camera_color_deques[index].size() >= 200:
            self.camera_color_deques[index].popleft()
        self.camera_color_deques[index].append(msg)

    def camera_depth_callback(self, index, msg):
        if self.camera_depth_deques[index].size() >= 200:
            self.camera_depth_deques[index].popleft()
        self.camera_depth_deques[index].append(msg)

    def camera_point_cloud_callback(self, index, msg):
        if self.camera_point_cloud_deques[index].size() >= 200:
            self.camera_point_cloud_deques[index].popleft()
        self.camera_point_cloud_deques[index].append(msg)

    def arm_joint_state_callback(self, index, msg):
        if self.arm_joint_state_deques[index].size() >= 200:
            self.arm_joint_state_deques[index].popleft()
        self.arm_joint_state_deques[index].append(msg)

    def arm_end_pose_callback(self, index, msg):
        if self.arm_end_pose_deques[index].size() >= 200:
            self.arm_end_pose_deques[index].popleft()
        self.arm_end_pose_deques[index].append(msg)

    def robot_base_vel_callback(self, index, msg):
        if self.robot_base_vel_deques[index].size() >= 200:
            self.robot_base_vel_deques[index].popleft()
        self.robot_base_vel_deques[index].append(msg)

    def camera_color_config_callback(self, index, msg):
        self.camera_color_intrinsics[index] = np.array(msg.k).reshape(3, 3)
        
        # 使用一次性定时器延迟执行TF查找
        def lookup_tf():
            try:
                # print(f"Looking up transform: {self.args.camera_color_parent_frame_ids[index]} -> {msg.header.frame_id}")
                # 检查变换是否可用（不使用阻塞的timeout）
                if self.tf_buffer.can_transform(
                    self.args.camera_color_parent_frame_ids[index], 
                    msg.header.frame_id, 
                    rclpy.time.Time()
                ):
                    transform = self.tf_buffer.lookup_transform(
                        self.args.camera_color_parent_frame_ids[index], 
                        msg.header.frame_id, 
                        rclpy.time.Time()
                    )
                    
                    trans = transform.transform.translation
                    rot = transform.transform.rotation
                    rot = euler_from_quaternion([rot.x, rot.y, rot.z, rot.w])
                    self.camera_color_extrinsics[index] = create_transformation_matrix(trans.x, trans.y, trans.z, rot[0], rot[1], rot[2])
                    # print(f"Successfully got camera_color transform {index}")
                else:
                    print(f"Transform not available for camera_color {index}, using identity")
                    self.camera_color_extrinsics[index] = np.eye(4)
                    
            except Exception as e:
                print(f'Failed to get camera_color transform {index}: {e}, using identity')
                self.camera_color_extrinsics[index] = np.eye(4)
            
            # 取消定时器并销毁订阅
            timer.cancel()
            self.destroy_subscription(self.camera_color_config_subscriber[index])
        
        # 创建一次性定时器
        timer = self.create_timer(2.0, lookup_tf)

    def camera_depth_config_callback(self, index, msg):
        self.camera_depth_intrinsics[index] = np.array(msg.k).reshape(3, 3)
        
        # 使用一次性定时器延迟执行TF查找
        def lookup_tf():
            try:
                # print(f"Looking up transform: {self.args.camera_depth_parent_frame_ids[index]} -> {msg.header.frame_id}")
                
                # 检查变换是否可用（不使用阻塞的timeout）
                if self.tf_buffer.can_transform(
                    self.args.camera_depth_parent_frame_ids[index], 
                    msg.header.frame_id, 
                    rclpy.time.Time()
                ):
                    transform = self.tf_buffer.lookup_transform(
                        self.args.camera_depth_parent_frame_ids[index], 
                        msg.header.frame_id, 
                        rclpy.time.Time()
                    )
                    
                    trans = transform.transform.translation
                    rot = transform.transform.rotation
                    rot = euler_from_quaternion([rot.x, rot.y, rot.z, rot.w])
                    self.camera_depth_extrinsics[index] = create_transformation_matrix(trans.x, trans.y, trans.z, rot[0], rot[1], rot[2])
                    # print(f"Successfully got camera_depth transform {index}")
                else:
                    print(f"Transform not available for camera_depth {index}, using identity")
                    self.camera_depth_extrinsics[index] = np.eye(4)
                    
            except Exception as e:
                print(f'Failed to get camera_depth transform {index}: {e}, using identity')
                self.camera_depth_extrinsics[index] = np.eye(4)
            
            # 取消定时器并销毁订阅
            timer.cancel()
            self.destroy_subscription(self.camera_depth_config_subscriber[index])
        
        # 创建一次性定时器
        timer = self.create_timer(2.0, lookup_tf)

    def camera_point_cloud_config_callback(self, index, msg):
        self.camera_point_cloud_intrinsics[index] = np.array(msg.k).reshape(3, 3)
        
        # 使用一次性定时器延迟执行TF查找
        def lookup_tf():
            try:
                # print(f"Looking up transform: {self.args.camera_point_cloud_parent_frame_ids[index]} -> {msg.header.frame_id}")
                
                # 检查变换是否可用（不使用阻塞的timeout）
                if self.tf_buffer.can_transform(
                    self.args.camera_point_cloud_parent_frame_ids[index], 
                    msg.header.frame_id, 
                    rclpy.time.Time()
                ):
                    transform = self.tf_buffer.lookup_transform(
                        self.args.camera_point_cloud_parent_frame_ids[index], 
                        msg.header.frame_id, 
                        rclpy.time.Time()
                    )
                    
                    trans = transform.transform.translation
                    rot = transform.transform.rotation
                    rot = euler_from_quaternion([rot.x, rot.y, rot.z, rot.w])
                    self.camera_point_cloud_extrinsics[index] = create_transformation_matrix(trans.x, trans.y, trans.z, rot[0], rot[1], rot[2])
                    # print(f"Successfully got camera_point_cloud transform {index}")
                else:
                    print(f"Transform not available for camera_point_cloud {index}, using identity")
                    self.camera_point_cloud_extrinsics[index] = np.eye(4)
                    
            except Exception as e:
                print(f'Failed to get camera_point_cloud transform {index}: {e}, using identity')
                self.camera_point_cloud_extrinsics[index] = np.eye(4)
            
            # 取消定时器并销毁订阅
            timer.cancel()
            self.destroy_subscription(self.camera_point_cloud_config_subscriber[index])
        
        # 创建一次性定时器
        timer = self.create_timer(2.0, lookup_tf)

    def interpolation_param(self, positions):
        positions = np.array(positions)
        # 构建矩阵A和向量b
        A = [np.ones_like(self.times)]
        for i in range(self.k - 1):
            A.append(self.times ** (i + 1))
        A = np.vstack(A).T
        b = positions
        # 解线性方程组得到多项式系数
        coeffs = np.linalg.solve(A, b)
        # 使用多项式系数计算给定时间的速度
        return coeffs

    def arm_joint_state_ctrl(self, joint_states):
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        joint_state_msg.name = [f'joint{i+1}' for i in range(self.args.arm_joint_state_dim)]
        self.last_ctrl_arm_joint_state = joint_states
        for i in range(len(joint_states)):
            joint_state_msg.position = [float(v) for v in joint_states[i]]
            self.arm_joint_state_ctrl_publisher[i].publish(joint_state_msg)

    def arm_joint_state_ctrl_interpolation(self, joint_states, is_new_action):
        if self.last_ctrl_arm_joint_state is None:
            last_ctrl_joint_state = np.concatenate(
                [np.array(self.arm_joint_state_deques[i].right().position) for i in range(len(self.args.arm_joint_state_names))], axis=0)
        else:
            last_ctrl_joint_state = np.concatenate(
                [np.array(joint_state) for joint_state in self.last_ctrl_arm_joint_state], axis=0)

        ctrl_joint_state = np.concatenate(
                [np.array(joint_state) for joint_state in joint_states], axis=0)
        joint_state_diff = ctrl_joint_state - last_ctrl_joint_state

        if is_new_action or max(joint_state_diff) > 0.5:
            append_to_history_list_step = 10
            hz = 500
            if max(joint_state_diff) > 0.5:
                step = int(max([max(abs(joint_state_diff[i*self.args.arm_joint_state_dim: (i+1)*self.args.arm_joint_state_dim-1]) / np.array(self.args.arm_steps_length[:self.args.arm_joint_state_dim-1])) for i in range(len(self.args.arm_joint_state_names))]))
            else:
                step = 50
            rate = self.create_rate(hz)
            joint_state_list = np.linspace(last_ctrl_joint_state, ctrl_joint_state, step + 1)
            for i in range(1, len(joint_state_list)):
                if self.arm_joint_state_ctrl_thread_return_lock.acquire(False):
                    return
                ctrl_joint_state = [joint_state_list[i][j*self.args.arm_joint_state_dim: (j+1)*self.args.arm_joint_state_dim] for j in range(len(self.args.arm_joint_state_names))]
                self.arm_joint_state_ctrl(ctrl_joint_state)
                if i % append_to_history_list_step == 0 or i + 1 == len(joint_state_list):
                    self.arm_joint_state_ctrl_history_list.append(ctrl_joint_state)
                rate.sleep()
            self.arm_joint_state_ctrl_history_list = self.arm_joint_state_ctrl_history_list[-self.k:]
            return

        if len(self.arm_joint_state_ctrl_history_list) == 0:
            for i in range(self.k):
                self.arm_joint_state_ctrl_history_list.append([last_ctrl_joint_state[j*self.args.arm_joint_state_dim: (j+1)*self.args.arm_joint_state_dim] for j in range(len(self.args.arm_joint_state_names))])
        self.arm_joint_state_ctrl_history_list.append(joint_states)
        self.arm_joint_state_ctrl_history_list = self.arm_joint_state_ctrl_history_list[-self.k:]
        coeffs = [self.interpolation_param([self.arm_joint_state_ctrl_history_list[k][i][j] for k in range(self.k)]) for i in range(len(self.args.arm_joint_state_names)) for j in range(self.args.arm_joint_state_dim)]
        hz = 500
        step = 10
        rate = self.create_rate(hz)
        for i in range(step):
            if self.arm_joint_state_ctrl_thread_return_lock.acquire(False):
                return
            ctrl_joint_state = [np.polyval(coeffs[j][::-1], (self.k - 2) + (i + 1) * (1.0 / step)) for j in range(len(coeffs))]
            self.arm_joint_state_ctrl([ctrl_joint_state[j*self.args.arm_joint_state_dim: (j+1)*self.args.arm_joint_state_dim] for j in range(len(self.args.arm_joint_state_names))])
            rate.sleep()

    def arm_joint_state_ctrl_linear_interpolation(self, joint_states, calc_step):
        if self.last_ctrl_arm_joint_state is None:
            last_ctrl_joint_state = np.concatenate(
                [np.array(self.arm_joint_state_deques[i].right().position) for i in range(len(self.args.arm_joint_state_names))], axis=0)
        else:
            last_ctrl_joint_state = np.concatenate(
                [np.array(joint_state) for joint_state in self.last_ctrl_arm_joint_state], axis=0)

        ctrl_joint_state = np.concatenate(
                [np.array(joint_state) for joint_state in joint_states], axis=0)
        joint_state_diff = ctrl_joint_state - last_ctrl_joint_state

        hz = 500
        if calc_step:
            step = int(max([max(abs(joint_state_diff[i*self.args.arm_joint_state_dim: (i+1)*self.args.arm_joint_state_dim-1]) / np.array(self.args.arm_steps_length[:self.args.arm_joint_state_dim-1])) for i in range(len(self.args.arm_joint_state_names))]))
            step = 1 if step == 0 else step
        else:
            step = 10
        rate = self.create_rate(hz)
        append_to_history_list_step = 10
        joint_state_list = np.linspace(last_ctrl_joint_state, ctrl_joint_state, step + 1)
        for i in range(1, len(joint_state_list)):
            if self.arm_joint_state_ctrl_thread_return_lock.acquire(False):
                return
            ctrl_joint_state = [joint_state_list[i][j * self.args.arm_joint_state_dim: (j + 1) * self.args.arm_joint_state_dim] for j in range(len(self.args.arm_joint_state_names))]
            self.arm_joint_state_ctrl(ctrl_joint_state)
            if i % append_to_history_list_step == 0 or i + 1 == len(joint_state_list):
                self.arm_joint_state_ctrl_history_list.append(ctrl_joint_state)
            rate.sleep()
        self.arm_joint_state_ctrl_history_list = self.arm_joint_state_ctrl_history_list[-self.k:]
        return

    def arm_joint_state_ctrl_interpolation_thread(self, joint_states, block, is_new_action):
        if self.arm_joint_state_ctrl_thread is not None:
            if self.args.preemptive_publishing and self.arm_joint_state_ctrl_thread_return_lock.locked():
                self.arm_joint_state_ctrl_thread_return_lock.release()
            self.arm_joint_state_ctrl_thread.join()
            self.arm_joint_state_ctrl_thread_return_lock.acquire(False)
            self.arm_joint_state_ctrl_thread = None
        self.arm_joint_state_ctrl_thread = threading.Thread(target=self.arm_joint_state_ctrl_interpolation,
                                                            args=(joint_states, is_new_action))
        self.arm_joint_state_ctrl_thread.start()
        if block:
            self.arm_joint_state_ctrl_thread.join()

    def arm_joint_state_ctrl_linear_interpolation_thread(self, joint_states, block, calc_step):
        if self.arm_joint_state_ctrl_thread is not None:
            if self.args.preemptive_publishing and self.arm_joint_state_ctrl_thread_return_lock.locked():
                self.arm_joint_state_ctrl_thread_return_lock.release()
            self.arm_joint_state_ctrl_thread.join()
            self.arm_joint_state_ctrl_thread_return_lock.acquire(False)
            self.arm_joint_state_ctrl_thread = None
        self.arm_joint_state_ctrl_thread = threading.Thread(target=self.arm_joint_state_ctrl_linear_interpolation,
                                                            args=(joint_states, calc_step))
        self.arm_joint_state_ctrl_thread.start()
        if block:
            self.arm_joint_state_ctrl_thread.join()

    def arm_end_pose_ctrl(self, end_poses):
        # print("left:", left)
        # print("right:", right)
        end_pose_msg = PoseStamped()
        end_pose_msg.header = Header()
        end_pose_msg.header.frame_id = "map"
        end_pose_msg.header.stamp = self.get_clock().now().to_msg()
        self.last_ctrl_arm_end_poses.append(end_poses)
        self.last_ctrl_arm_end_poses = self.last_ctrl_arm_end_poses[-self.args.obs_history_num:]
        for i in range(len(end_poses)):
            end_pose_msg.pose.position.x = end_poses[i][0]
            end_pose_msg.pose.position.y = end_poses[i][1]
            end_pose_msg.pose.position.z = end_poses[i][2]
            # q = quaternion_from_euler(end_poses[i][3], end_poses[i][4], end_poses[i][5])
            # end_pose_msg.pose.orientation.x = q[0]
            # end_pose_msg.pose.orientation.y = q[1]
            # end_pose_msg.pose.orientation.z = q[2]
            # end_pose_msg.pose.orientation.w = q[3]
            end_pose_msg.pose.orientation.x = end_poses[i][3]
            end_pose_msg.pose.orientation.y = end_poses[i][4]
            end_pose_msg.pose.orientation.z = end_poses[i][5]
            end_pose_msg.pose.orientation.w = end_poses[i][6]
            end_pose_msg.pose.orientation.w += self.args.gripper_offset[i]
            self.arm_end_pose_ctrl_publisher[i].publish(end_pose_msg)

    def arm_end_pose_ctrl_linear_interpolation(self, end_poses, calc_step):
        if len(self.last_ctrl_arm_end_poses) == 0:
            last_ctrl_end_pose = np.concatenate([
                np.array([self.arm_end_pose_deques[i].right().pose.position.x, self.arm_end_pose_deques[i].right().pose.position.y,
                          self.arm_end_pose_deques[i].right().pose.position.z,
                          self.arm_end_pose_deques[i].right().pose.orientation.x, self.arm_end_pose_deques[i].right().pose.orientation.y,
                          self.arm_end_pose_deques[i].right().pose.orientation.z, self.arm_end_pose_deques[i].right().pose.orientation.w])
                for i in range(len(self.args.arm_end_pose_names))], axis=0)
        else:
            last_ctrl_end_pose = np.concatenate(
                [np.array(end_poses) for end_poses in self.last_ctrl_arm_end_poses[-1]], axis=0)

        hz = 500
        if calc_step:
            ctrl_end_pose = np.concatenate([np.array(end_pose) for end_pose in end_poses], axis=0)
            end_pose_diff = ctrl_end_pose - last_ctrl_end_pose

            step_position = int(max([max(abs(end_pose_diff[i*self.args.arm_end_pose_dim: i*self.args.arm_end_pose_dim+3]) / np.array(self.args.arm_steps_length[:3])) for i in range(len(self.args.arm_end_pose_names))]))
            step_grasp = int(max([abs(end_pose_diff[(i+1)*self.args.arm_end_pose_dim-1]) / np.array(self.args.arm_steps_length[self.args.arm_end_pose_dim-1]) for i in range(len(self.args.arm_end_pose_names))]))
            step = max([step_grasp, step_position])
        else:
            step = 10
        rate = self.create_rate(hz)

        ctrl_traj_xyzg = []
        ctrl_traj_xyzw = []
        for i in range(len(self.args.arm_end_pose_names)):
            ctrl_end_pose_xyzg = [end_poses[i][0], end_poses[i][1], end_poses[i][2], end_poses[i][6]]
            ctrl_end_pose_xyzw = quaternion_from_euler(end_poses[i][3], end_poses[i][4], end_poses[i][5])
            last_ctrl_end_pose_xyzg = [last_ctrl_end_pose[i*self.args.arm_end_pose_dim+0], last_ctrl_end_pose[i*self.args.arm_end_pose_dim+1], last_ctrl_end_pose[i*self.args.arm_end_pose_dim+2], last_ctrl_end_pose[i*self.args.arm_end_pose_dim+6]]
            last_ctrl_end_pose_xyzw = quaternion_from_euler(last_ctrl_end_pose[i*self.args.arm_end_pose_dim+3], last_ctrl_end_pose[i*self.args.arm_end_pose_dim+4], last_ctrl_end_pose[i*self.args.arm_end_pose_dim+5])
            traj_xyzg = np.linspace(last_ctrl_end_pose_xyzg, ctrl_end_pose_xyzg, step + 1)[1:]
            traj_xyzw = [geometric_slerp(np.array(last_ctrl_end_pose_xyzw), np.array(ctrl_end_pose_xyzw), (j+1) / step) for j in range(step)]
            ctrl_traj_xyzg.append(traj_xyzg)
            ctrl_traj_xyzw.append(traj_xyzw)
        for i in range(step):
            if self.arm_end_pose_ctrl_thread_return_lock.acquire(False):
                return
            ctrl_end_poses = []
            for j in range(len(self.args.arm_end_pose_names)):
                ctrl_rpy = euler_from_quaternion([ctrl_traj_xyzw[j][i][0], ctrl_traj_xyzw[j][i][1], ctrl_traj_xyzw[j][i][2], ctrl_traj_xyzw[j][i][3]])
                ctrl_end_pose = [ctrl_traj_xyzg[j][i][0], ctrl_traj_xyzg[j][i][1], ctrl_traj_xyzg[j][i][2],
                                 ctrl_rpy[0], ctrl_rpy[1], ctrl_rpy[2],
                                 ctrl_traj_xyzg[j][i][3]]
                ctrl_end_poses.append(ctrl_end_pose)
            self.arm_end_pose_ctrl(ctrl_end_poses)
            rate.sleep()

    def arm_end_pose_ctrl_linear_interpolation_thread(self, end_poses, block, calc_step):
        if self.arm_end_pose_ctrl_thread is not None:
            if self.args.preemptive_publishing and self.arm_end_pose_ctrl_thread_return_lock.locked():
                self.arm_end_pose_ctrl_thread_return_lock.release()
            self.arm_end_pose_ctrl_thread.join()
            self.arm_end_pose_ctrl_thread_return_lock.acquire(False)
            self.arm_end_pose_ctrl_thread = None
        self.arm_end_pose_ctrl_thread = threading.Thread(target=self.arm_end_pose_ctrl_linear_interpolation,
                                                         args=(end_poses, calc_step))
        self.arm_end_pose_ctrl_thread.start()
        if block:
            self.arm_end_pose_ctrl_thread.join()

    def robot_base_vel_ctrl(self, vel):
        vel_msg = Twist()
        vel_msg.linear.x = vel[0]
        vel_msg.linear.y = vel[1]
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = vel[2]
        self.robot_base_vel_ctrl_publisher.publish(vel_msg)

    def get_frame(self):
        rate = self.create_rate(1)
        while rclpy.ok() and not self.all_config_exist:
            for i in range(len(self.camera_color_intrinsics)):
                self.camera_color_intrinsics[i] = np.eye(3)
            for i in range(len(self.camera_depth_intrinsics)):
                self.camera_depth_intrinsics[i] = np.eye(3)
            for i in range(len(self.camera_point_cloud_intrinsics)):
                self.camera_point_cloud_intrinsics[i] = np.eye(3)
            for i in range(len(self.camera_color_extrinsics)):
                self.camera_color_extrinsics[i] = np.eye(4)
            for i in range(len(self.camera_depth_extrinsics)):
                self.camera_depth_extrinsics[i] = np.eye(4)
            for i in range(len(self.camera_point_cloud_extrinsics)):
                self.camera_point_cloud_extrinsics[i] = np.eye(4)
            
            self.all_config_exist = True
            if any(item is None for item in self.camera_color_intrinsics):
                self.all_config_exist = False
                print("camera color intrinsic config not exist")
            if any(item is None for item in self.camera_depth_intrinsics):
                self.all_config_exist = False
                print("camera depth intrinsic config not exist")
            if not self.args.use_camera_color_depth_to_point_cloud and any(item is None for item in self.camera_point_cloud_intrinsics):
                self.all_config_exist = False
                print("camera point cloud intrinsic config not exist")
            if any(item is None for item in self.camera_color_extrinsics):
                self.all_config_exist = False
                print("camera color extrinsic config not exist")
            if any(item is None for item in self.camera_depth_extrinsics):
                self.all_config_exist = False
                print("camera depth extrinsic config not exist")
            if not self.args.use_camera_color_depth_to_point_cloud and any(item is None for item in self.camera_point_cloud_extrinsics):
                self.all_config_exist = False
                print("camera point cloud extrinsic config not exist")
            rate.sleep()

        camera_colors = [self.camera_color_deques[i].right() for i in range(len(self.args.camera_color_names))]
        camera_depths = [self.camera_depth_deques[i].right() for i in range(len(self.args.camera_depth_names))]
        camera_point_clouds = [self.camera_point_cloud_deques[i].right() for i in range(len(self.args.camera_point_cloud_names))] if not self.args.use_camera_color_depth_to_point_cloud else []
        arm_joint_states = [self.arm_joint_state_deques[i].right() for i in range(len(self.args.arm_joint_state_names))]
        arm_end_poses = [self.arm_end_pose_deques[i].right() for i in range(len(self.args.arm_end_pose_names))]
        robot_base_vels = [self.robot_base_vel_deques[i].right() for i in range(len(self.args.robot_base_vel_names))]
        frame_time = max([rclpy.time.Time.from_msg(msg.header.stamp).nanoseconds / 1e9 for msg in
                          (camera_colors + camera_depths +camera_point_clouds + arm_joint_states +
                           arm_end_poses + robot_base_vels)])
        for i in range(len(self.args.camera_color_names)):
            closer_time_diff = math.inf
            while (self.camera_color_deques[i].size() > 0 and
                   abs(rclpy.time.Time.from_msg(self.camera_color_deques[i].left().header.stamp).nanoseconds / 1e9 - frame_time) < closer_time_diff):
                closer_time_diff = abs(rclpy.time.Time.from_msg(self.camera_color_deques[i].left().header.stamp).nanoseconds / 1e9 - frame_time)
                camera_colors[i] = self.camera_color_deques[i].popleft()
        for i in range(len(self.args.camera_depth_names)):
            closer_time_diff = math.inf
            while (self.camera_depth_deques[i].size() > 0 and
                   abs(rclpy.time.Time.from_msg(self.camera_depth_deques[i].left().header.stamp).nanoseconds / 1e9 - frame_time) < closer_time_diff):
                closer_time_diff = abs(rclpy.time.Time.from_msg(self.camera_depth_deques[i].left().header.stamp).nanoseconds / 1e9 - frame_time)
                camera_depths[i] = self.camera_depth_deques[i].popleft()
        if not self.args.use_camera_color_depth_to_point_cloud:
            for i in range(len(self.args.camera_point_cloud_names)):
                closer_time_diff = math.inf
                while (self.camera_point_cloud_deques[i].size() > 0 and
                    abs(rclpy.time.Time.from_msg(self.camera_point_cloud_deques[i].left().header.stamp).nanoseconds / 1e9 - frame_time) < closer_time_diff):
                    closer_time_diff = abs(rclpy.time.Time.from_msg(self.camera_point_cloud_deques[i].left().header.stamp).nanoseconds / 1e9 - frame_time)
                    camera_point_clouds[i] = self.camera_point_cloud_deques[i].popleft()
        for i in range(len(self.args.arm_joint_state_names)):
            closer_time_diff = math.inf
            while (self.arm_joint_state_deques[i].size() > 0 and
                   abs(rclpy.time.Time.from_msg(self.arm_joint_state_deques[i].left().header.stamp).nanoseconds / 1e9 - frame_time) < closer_time_diff):
                closer_time_diff = abs(rclpy.time.Time.from_msg(self.arm_joint_state_deques[i].left().header.stamp).nanoseconds / 1e9 - frame_time)
                arm_joint_states[i] = self.arm_joint_state_deques[i].popleft()
        for i in range(len(self.args.arm_end_pose_names)):
            closer_time_diff = math.inf
            while (self.arm_end_pose_deques[i].size() > 0 and
                   abs(rclpy.time.Time.from_msg(self.arm_end_pose_deques[i].left().header.stamp).nanoseconds / 1e9 - frame_time) < closer_time_diff):
                closer_time_diff = abs(rclpy.time.Time.from_msg(self.arm_end_pose_deques[i].left().header.stamp).nanoseconds / 1e9 - frame_time)
                arm_end_poses[i] = self.arm_end_pose_deques[i].popleft()
        for i in range(len(self.args.robot_base_vel_names)):
            closer_time_diff = math.inf
            while (self.robot_base_vel_deques[i].size() > 0 and
                   abs(rclpy.time.Time.from_msg(self.robot_base_vel_deques[i].left().header.stamp).nanoseconds / 1e9 - frame_time) < closer_time_diff):
                closer_time_diff = abs(rclpy.time.Time.from_msg(self.robot_base_vel_deques[i].left().header.stamp).nanoseconds / 1e9 - frame_time)
                robot_base_vels[i] = self.robot_base_vel_deques[i].popleft()

        # for i in range(len(self.args.camera_color_names)):
        #     while rclpy.time.Time.from_msg(self.camera_color_deques[i].left().header.stamp).nanoseconds / 1e9 < frame_time:
        #         self.camera_color_deques[i].popleft()
        #     camera_colors[i] = self.camera_color_deques[i].popleft()
        # for i in range(len(self.args.camera_depth_names)):
        #     while rclpy.time.Time.from_msg(self.camera_depth_deques[i].left().header.stamp).nanoseconds / 1e9 < frame_time:
        #         self.camera_depth_deques[i].popleft()
        #     camera_depths[i] = self.camera_depth_deques[i].popleft()
        # if not self.args.use_camera_color_depth_to_point_cloud:
        #     for i in range(len(self.args.camera_point_cloud_names)):
        #         while rclpy.time.Time.from_msg(self.camera_point_cloud_deques[i].left().header.stamp).nanoseconds / 1e9 < frame_time:
        #             self.camera_point_cloud_deques[i].popleft()
        #         camera_point_clouds[i] = self.camera_point_cloud_deques[i].popleft()
        # for i in range(len(self.args.arm_joint_state_names)):
        #     while rclpy.time.Time.from_msg(self.arm_joint_state_deques[i].left().header.stamp).nanoseconds / 1e9 < frame_time:
        #         self.arm_joint_state_deques[i].popleft()
        #     arm_joint_states[i] = self.arm_joint_state_deques[i].popleft()
        # for i in range(len(self.args.arm_end_pose_names)):
        #     while rclpy.time.Time.from_msg(self.arm_end_pose_deques[i].left().header.stamp).nanoseconds / 1e9 < frame_time:
        #         self.arm_end_pose_deques[i].popleft()
        #     arm_end_poses[i] = self.arm_end_pose_deques[i].popleft()
        # for i in range(len(self.args.robot_base_vel_names)):
        #     while rclpy.time.Time.from_msg(self.robot_base_vel_deques[i].left().header.stamp).nanoseconds / 1e9 < frame_time:
        #         self.robot_base_vel_deques[i].popleft()
        #     robot_base_vels[i] = self.robot_base_vel_deques[i].popleft()

        if len(self.camera_color_history_list) == 0:
            for i in range(self.args.obs_history_num):
                self.camera_color_history_list.append(camera_colors)
        self.camera_color_history_list.append(camera_colors)
        self.camera_color_history_list = self.camera_color_history_list[-self.args.obs_history_num:]

        if len(self.camera_depth_history_list) == 0:
            for i in range(self.args.obs_history_num):
                self.camera_depth_history_list.append(camera_depths)
        self.camera_depth_history_list.append(camera_depths)
        self.camera_depth_history_list = self.camera_depth_history_list[-self.args.obs_history_num:]

        if len(self.camera_point_cloud_history_list) == 0:
            for i in range(self.args.obs_history_num):
                self.camera_point_cloud_history_list.append(camera_point_clouds)
        self.camera_point_cloud_history_list.append(camera_point_clouds)
        self.camera_point_cloud_history_list = self.camera_point_cloud_history_list[-self.args.obs_history_num:]

        if len(self.arm_joint_state_history_list) == 0:
            for i in range(self.args.obs_history_num):
                self.arm_joint_state_history_list.append(arm_joint_states)
        self.arm_joint_state_history_list.append(arm_joint_states)
        self.arm_joint_state_history_list = self.arm_joint_state_history_list[-self.args.obs_history_num:]

        if len(self.arm_end_pose_history_list) == 0:
            for i in range(self.args.obs_history_num):
                self.arm_end_pose_history_list.append(arm_end_poses)
        self.arm_end_pose_history_list.append(arm_end_poses)
        self.arm_end_pose_history_list = self.arm_end_pose_history_list[-self.args.obs_history_num:]

        if len(self.robot_base_vel_history_list) == 0:
            for i in range(self.args.obs_history_num):
                self.robot_base_vel_history_list.append(robot_base_vels)
        self.robot_base_vel_history_list.append(robot_base_vels)
        self.robot_base_vel_history_list = self.robot_base_vel_history_list[-self.args.obs_history_num:]

        return (self.instruction,
                self.camera_color_history_list, self.camera_depth_history_list, self.camera_point_cloud_history_list,
                self.arm_joint_state_history_list, self.arm_end_pose_history_list,
                self.robot_base_vel_history_list, self.last_ctrl_arm_end_poses)

    def check_frame(self):
        for i in range(len(self.args.camera_color_names)):
            if self.camera_color_deques[i].size() == 0:
                print(self.args.camera_color_topics[i], "has no data")
                return False
        for i in range(len(self.args.camera_depth_names)):
            if self.camera_depth_deques[i].size() == 0:
                print(self.args.camera_depth_topics[i], "has no data")
                return False
        if not self.args.use_camera_color_depth_to_point_cloud:
            for i in range(len(self.args.camera_point_cloud_names)):
                if self.camera_point_cloud_deques[i].size() == 0:
                    print(self.args.camera_point_cloud_topics[i], "has no data")
                    return False
        for i in range(len(self.args.arm_joint_state_names)):
            if self.arm_joint_state_deques[i].size() == 0:
                print(self.args.arm_joint_state_topics[i], "has no data")
                return False
        for i in range(len(self.args.arm_end_pose_names)):
            if self.arm_end_pose_deques[i].size() == 0:
                print(self.args.arm_end_pose_topics[i], "has no data")
                return False
        for i in range(len(self.args.robot_base_vel_names)):
            if self.robot_base_vel_deques[i].size() == 0:
                print(self.args.robot_base_vel_topics[i], "has no data")
                return False
        return True

    def change_inference_status(self, request, response):
        self.inference_status = request.status
        return response

    def get_inference_status(self):
        return self.inference_status

    def set_inference_status(self, status):
        self.inference_status = status


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_publish_step', action='store', type=int, help='max_publish_step', default=10000,
                        required=False)

    # 模型相关参数
    parser.add_argument('--pretrained_model_name_or_path', type=str, required=True, 
                        help='Name or path to the pretrained model')
    parser.add_argument('--stats_path', type=str, required=False, default=None,
                        help='Path to custom stats.json file (optional, overrides model default)')
    parser.add_argument('--ctrl_freq', action='store', type=int, 
                        help='The control frequency of the robot',
                        default=25, required=False)
    # parser.add_argument('--lang_embeddings_path', type=str, required=True, 
    #                     help='Path to the pre-encoded language instruction embeddings')

    parser.add_argument('--instruction_topic', action='store', type=str, help='instruction_topic',
                        default='/instruction', required=False)

    parser.add_argument('--camera_color_names', action='store', type=str, help='camera_color_names',
                        default=['left', 'front', 'right'],
                        required=False)
    parser.add_argument('--camera_color_parent_frame_ids', action='store', type=str, help='camera_color_parent_frame_ids',
                        default=['camera_l_link', 'camera_f_link', 'camera_r_link'],
                        required=False)
    parser.add_argument('--camera_color_topics', action='store', type=str, help='camera_color_topics',
                        default=['/camera_l/color/image_raw', '/camera_f/color/image_raw', '/camera_r/color/image_raw'],
                        required=False)
    parser.add_argument('--camera_color_config_topics', action='store', type=str, help='camera_color_config_topics',
                        default=['/camera_l/color/camera_info', '/camera_f/color/camera_info', '/camera_r/color/camera_info'],
                        required=False)
    parser.add_argument('--camera_depth_names', action='store', type=str, help='camera_depth_names',
                        default=['left', 'front', 'right'],
                        required=False)
    parser.add_argument('--camera_depth_parent_frame_ids', action='store', type=str, help='camera_depth_parent_frame_ids',
                        default=['camera_l_link', 'camera_f_link', 'camera_r_link'],
                        required=False)
    parser.add_argument('--camera_depth_topics', action='store', type=str, help='camera_depth_topics',
                        default=['/camera_l/aligned_depth_to_color/image_raw', '/camera_f/aligned_depth_to_color/image_raw', '/camera_r/aligned_depth_to_color/image_raw'],
                        required=False)
    parser.add_argument('--camera_depth_config_topics', action='store', type=str, help='camera_depth_config_topics',
                        default=['/camera_l/aligned_depth_to_color/camera_info', '/camera_f/aligned_depth_to_color/camera_info', '/camera_r/aligned_depth_to_color/camera_info'],
                        required=False)
    parser.add_argument('--use_camera_color_depth_to_point_cloud', action='store', type=bool, help='use_camera_color_depth_to_point_cloud',
                        default=True,
                        required=False)
    parser.add_argument('--camera_point_cloud_names', action='store', type=str, help='camera_point_cloud_names',
                        default=['left', 'front', 'right'],
                        required=False)
    parser.add_argument('--camera_point_cloud_parent_frame_ids', action='store', type=str, help='camera_point_cloud_parent_frame_ids',
                        default=['camera_l_link', 'camera_f_link', 'camera_r_link'],
                        required=False)
    parser.add_argument('--camera_point_cloud_topics', action='store', type=str, help='camera_point_cloud_topics',
                        default=['/camera_l/depth/color/points', '/camera_f/depth/color/points', '/camera_r/depth/color/points'],
                        required=False)
    parser.add_argument('--camera_point_cloud_config_topics', action='store', type=str, help='camera_point_cloud_config_topics',
                        default=['/camera_l/aligned_depth_to_color/camera_info', '/camera_f/aligned_depth_to_color/camera_info', '/camera_r/aligned_depth_to_color/camera_info'],
                        required=False)
    parser.add_argument('--arm_joint_state_names', action='store', type=str, help='arm_joint_state_names',
                        default=['left', 'right'],
                        required=False)
    parser.add_argument('--arm_joint_state_topics', action='store', type=str, help='arm_joint_state_topics',
                        default=['/puppet/joint_left', '/puppet/joint_right'],
                        required=False)
    parser.add_argument('--arm_end_pose_names', action='store', type=str, help='arm_end_pose_names',
                        default=[],
                        required=False)
    parser.add_argument('--arm_end_pose_topics', action='store', type=str, help='arm_end_pose_topics',
                        default=[],
                        required=False)
    parser.add_argument('--robot_base_vel_names', action='store', type=str, help='robot_base_vel_names',
                        default=[],
                        required=False)
    parser.add_argument('--robot_base_vel_topics', action='store', type=str, help='robot_base_vel_topics',
                        default=[],
                        required=False)
    parser.add_argument('--arm_joint_state_ctrl_topics', action='store', type=str, help='arm_joint_state_ctrl_topics',
                        default=['/joint_left_states', '/joint_right_states'],
                        required=False)
    parser.add_argument('--arm_end_pose_ctrl_topics', action='store', type=str, help='arm_end_pose_ctrl_topics',
                        default=[],
                        required=False)
    parser.add_argument('--robot_base_vel_ctrl_topic', action='store', type=str, help='robot_base_vel_ctrl_topic',
                        default='/cmd_vel',
                        required=False)
    parser.add_argument('--gripper_offset', nargs='+', action='store', type=float, help='gripper_offset', default=[0], required=False)

    parser.add_argument('--use_camera_color', action='store', type=bool, help='use_camera_color', default=True, required=False)
    parser.add_argument('--use_camera_depth', action='store', type=bool, help='use_camera_depth', default=False, required=False)
    parser.add_argument('--use_camera_point_cloud', action='store', type=bool, help='use_camera_point_cloud', default=False, required=False)
    parser.add_argument('--use_camera_point_cloud_rgb', action='store', type=bool, help='use_camera_point_cloud_rgb', default=True, required=False)
    parser.add_argument('--use_robot_base', action='store', type=int, help='use_robot_base', default=0, required=False)
    parser.add_argument('--robot_base_dim', action='store', type=int, help='robot_base_dim', default=3, required=False)
    parser.add_argument('--use_arm_joint_state', action='store', type=int, help='use_arm_joint_state', default=3, required=False)
    parser.add_argument('--arm_joint_state_dim', action='store', type=int, help='arm_joint_state_dim', default=7, required=False)
    parser.add_argument('--use_arm_end_pose', action='store', type=int, help='use_arm_end_pose', default=0, required=False)
    parser.add_argument('--arm_end_pose_dim', action='store', type=int, help='arm_end_pose_dim', default=7, required=False)

    parser.add_argument('--obs_history_num', action='store', type=int, help='obs_history_num', default=1, required=False)
    parser.add_argument('--use_instruction', action='store', type=bool, help='use_instruction', default=False, required=False)
    parser.add_argument('--instruction', action='store', type=str, help='instruction',
                        default='null', required=False)
    parser.add_argument('--camera_point_cloud_point_num', action='store', type=int, help='camera_point_cloud_point_num', default=5000, required=False)
    parser.add_argument('--camera_point_cloud_voxel_size', action='store', type=float, help='camera_point_cloud_voxel_size', default=0.01, required=False)
    parser.add_argument('--use_farthest_point_down_sample', action='store', type=bool, help='use_farthest_point_down_sample', default=False, required=False)

    parser.add_argument('--aloha_inference_status_service', action='store', type=str,
                        help='aloha_inference_status_service',
                        default='/aloha/inference_status_service', required=False)
    parser.add_argument('--pos_lookahead_step', action='store', type=int, help='pos_lookahead_step',
                        default=50, required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size',
                        default=50, required=False)
    parser.add_argument('--arm_steps_length', action='store', type=float, help='arm_steps_length',
                        default=[0.0015, 0.0015, 0.0015, 0.0015, 0.0015, 0.0015, 0.03], required=False)
    parser.add_argument('--robot_base_steps_length', action='store', type=float, help='robot_base_steps_length',
                        default=[0.1, 0.1], required=False)
    parser.add_argument('--asynchronous_inference', action='store', type=bool, help='asynchronous_inference',
                        default=True, required=False)
    parser.add_argument('--preemptive_publishing', action='store', type=bool, help='preemptive_publishing',
                        default=False, required=False)
    parser.add_argument('--blocking_publish', action='store', type=bool, help='blocking_publish',
                        default=True, required=False)

    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    rclpy.init()
    ros_operator = RosOperator(args)
    rclpy.spin(ros_operator)


if __name__ == '__main__':
    main()

