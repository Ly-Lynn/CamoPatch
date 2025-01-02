# CamoPatch
 Code for our paper "CamoPatch: An Evolutionary Strategy for Generating Camoflauged Adversarial Patches" published at NeurIPS 2023

# Installation
1. Clone this repo

```bash
git clone --recurse-submodules https://Ly-Lynn:$github_token@github.com/Ly-Lynn/CamoPatch.git
```

2. Create folder contains the pretrained models, and move the pretrained models to this folder

```bash
mkdir -p /CamoPatch/Face_Recognition_Resource/pretrained_model
```

3. Dependencies
```bash
pip install onnxruntime insightface
```

4. (optinal) Create output folders
```bash
mkdir -p /CamoPatch/outputs
```
5. Run!
```bash
python ConCamoPatch.py --model_name  restnet_vggface --queries 200000 --image1_dir D:\codePJ\RESEARCH\GECCO2025\lfw_dataset\lfw_crop_margin_5\lfw_crop_margin_5\Aaron_Peirsol\Aaron_Peirsol_0001.jpg --image2_dir D:\codePJ\RESEARCH\GECCO2025\lfw_dataset\lfw_crop_margin_5\lfw_crop_margin_5\Aaron_Peirsol\Aaron_Peirsol_0002.jpg --true_label 0 --save_directory outputs
```
