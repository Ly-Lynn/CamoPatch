# CamoPatch

## Project Overview

*CamoPatch* is a reimplementation and extension of the original [CamoPatch project](https://github.com/phoenixwilliams/CamoPatch). This version includes an additional feature designed for launching adversarial attacks on face verification models and fix some errors from the original.

## Installation Guide

Follow the steps below to set up and run CamoPatch:

### 1. Clone the Repository
Begin by cloning this repository with its submodules:

```bash
git clone --recurse-submodules https://Ly-Lynn:$github_token@github.com/Ly-Lynn/CamoPatch.git
```

### 2. Prepare Pretrained Model Directory
Create a directory to store the pretrained models and move them into this folder:

```bash
mkdir -p /CamoPatch/Face_Recognition_Resource/pretrained_model
```

### 3. Install Dependencies
Ensure you have the necessary dependencies installed:

```bash
pip install onnxruntime insightface
```

### 4. (Optional) Create Output Folders
Create directories for storing output data:

```bash
mkdir -p /CamoPatch/outputs
```

### 5. Run the Project
Execute the script using the following command:

```bash
python ConCamoPatch.py --model_name restnet_vggface --queries 200000 --image1_dir <image_1 directory> --image2_dir <image_2 directory> --true_label 0 --save_directory outputs
```

Replace `<image_1 directory>` and `<image_2 directory>` with the paths to the images you wish to use.
