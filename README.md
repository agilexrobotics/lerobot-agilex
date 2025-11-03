# LeRobot
Referencing https://github.com/huggingface/lerobot.git, For more details, please refer to this link.
Data collection method reference: https://github.com/agilexrobotics/data_tools
## Installation

LeRobot works with Python 3.10+ and PyTorch 2.2+.

### Environment Setup

Create a virtual environment with Python 3.10 and activate it, e.g. with [`miniconda`](https://docs.anaconda.com/free/miniconda/index.html):

```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
```

When using `miniconda`, install `ffmpeg` in your environment:

```bash
conda install ffmpeg -c conda-forge
```

> **NOTE:** This usually installs `ffmpeg 7.X` for your platform compiled with the `libsvtav1` encoder. If `libsvtav1` is not supported (check supported encoders with `ffmpeg -encoders`), you can:
>
> - _[On any platform]_ Explicitly install `ffmpeg 7.X` using:
>
> ```bash
> conda install ffmpeg=7.1.1 -c conda-forge
> ```
>
> - _[On Linux only]_ Install [ffmpeg build dependencies](https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu#GettheDependencies) and [compile ffmpeg from source with libsvtav1](https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu#libsvtav1), and make sure you use the corresponding ffmpeg binary to your install with `which ffmpeg`.

### Install LeRobot 🤗

#### From Source

First, clone the repository and navigate into the directory:

```bash
git clone https://github.com/agilexrobotics/lerobot-agilex.git lerobot
cd lerobot
```

Then, install the library in editable mode. This is useful if you plan to contribute to the code.

```bash
pip install -e .
```
### Weights & Biases

To use [Weights and Biases](https://docs.wandb.ai/quickstart) for experiment tracking, log in with

```bash
wandb login
```

(note: you will also need to enable WandB in the configuration. See below.)

## Visualize datasets

Check out [example 1](https://github.com/huggingface/lerobot/blob/main/examples/1_load_lerobot_dataset.py) that illustrates how to use our dataset class which automatically downloads data from the Hugging Face hub.

You can also locally visualize episodes from a dataset on the hub by executing our script from the command line:

```bash
python -m lerobot.scripts.visualize_dataset \
    --repo-id lerobot/pusht \
    --episode-index 0
```

or from a dataset in a local folder with the `root` option and the `--local-files-only` (in the following case the dataset will be searched for in `./my_local_data_dir/lerobot/pusht`)

```bash
python -m lerobot.scripts.visualize_dataset \
    --repo-id lerobot/pusht \
    --root ./my_local_data_dir \
    --local-files-only 1 \
    --episode-index 0
```

It will open `rerun.io` and display the camera streams, robot states and actions, like this:

https://github-production-user-asset-6210df.s3.amazonaws.com/4681518/328035972-fd46b787-b532-47e2-bb6f-fd536a55a7ed.mov?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240505%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240505T172924Z&X-Amz-Expires=300&X-Amz-Signature=d680b26c532eeaf80740f08af3320d22ad0b8a4e4da1bcc4f33142c15b509eda&X-Amz-SignedHeaders=host&actor_id=24889239&key_id=0&repo_id=748713144

Our script can also visualize datasets stored on a distant server. See `python -m lerobot.scripts.visualize_dataset --help` for more instructions.



## Train
```bash
conda activate lerobot && cd ~/lerobot && python src/lerobot/scripts/train.py --dataset.repo_id=data --policy.type=act --output_dir=/home/agilex/checkpoint_lerobot --job_name=data --policy.device=cuda --wandb.enable=false --dataset.root=/home/agilex/data_lerobot/ --policy.repo_id=data --batch_size 1 --eval.batch_size 1
```

## inference
```bash
conda activate lerobot && cd ~/lerobot/scripts && python lerobot_inference-ros2.py --policy act --pretrained_model_name_or_path ~/checkpoint_lerobot/checkpoints/100000/pretrained_model/
```
