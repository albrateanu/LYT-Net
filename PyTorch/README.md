# LYT-Net: Lightweight YUV Transformer-based Network for Low-Light Image Enhancement

<div align="center">
  
![Logo](../figs/Logo.png)

[![arXiv](https://img.shields.io/badge/arxiv-paper-179bd3)](https://arxiv.org/abs/2401.15204)
</div>

## Description
This is the PyTorch version of LYT-Net.

**Note:** This is not the implementation used for reporting paper results.
## Experiment

### 1. Create Environment
- Make Conda Environment
```bash
conda create -n LYT_Torch python=3.9 -y
conda activate LYT_Torch
```
- Install Dependencies
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm tensorboard

pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips thop timm
```

### 2. Prepare Datasets
Download the LOLv1 and LOLv2 datasets:

LOLv1 - [Google Drive](https://drive.google.com/file/d/1vhJg75hIpYvsmryyaxdygAWeHuiY_HWu/view?usp=sharing)

LOLv2 - [Google Drive](https://drive.google.com/file/d/1OMfP6Ks2QKJcru1wS2eP629PgvKqF2Tw/view?usp=sharing)

**Note:** Under the main directory, create a folder called ```data``` and place the dataset folders inside it.
<details>
  <summary>
  <b>Datasets should be organized as follows:</b>
  </summary>

  ```
    |--data   
    |    |--LOLv1
    |    |    |--Train
    |    |    |    |--input
    |    |    |    |     ...
    |    |    |    |--target
    |    |    |    |     ...
    |    |    |--Test
    |    |    |    |--input
    |    |    |    |     ...
    |    |    |    |--target
    |    |    |    |     ...
    |    |--LOLv2
    |    |    |--Real_captured
    |    |    |    |--Train
    |    |    |    |    |--Low
    |    |    |    |    |     ...
    |    |    |    |    |--Normal
    |    |    |    |    |     ...
    |    |    |    |--Test
    |    |    |    |    |--Low
    |    |    |    |    |     ...
    |    |    |    |    |--Normal
    |    |    |    |    |     ...
    |    |    |--Synthetic
    |    |    |    |--Train
    |    |    |    |    |--Low
    |    |    |    |    |    ...
    |    |    |    |    |--Normal
    |    |    |    |    |    ...
    |    |    |    |--Test
    |    |    |    |    |--Low
    |    |    |    |    |    ...
    |    |    |    |    |--Normal
    |    |    |    |    |    ...
  ```

</details>

**Note:** ```data``` directory should be placed under the ```PyTorch``` implementation folder.

### 3. Test
You can test the model using the following commands. Pre-trained weights are available at [Google Drive](https://drive.google.com/file/d/1GeEkasO2ubFi847pzrxfQ1fB3Y9NuhZ1/view?usp=sharing). GT Mean evaluation is enabled by default and can be deactivated by setting the boolean flag ```gt_mean=False``` in the ```compute_psnr()``` method under the ```test.py``` file.

```bash
python test.py
```

**Note:** Please modify the dataset paths in ```test.py``` as per your requirements.

### 4. Compute Complexity
You can test the model complexity (FLOPS/Params) using the following command:
```bash
# To run FLOPS check with default (1,256,256,3)
python macs.py
```

### 5. Train
You can train the model using the following command:

```bash
python train.py
```

**Note:** Please modify the dataset paths in ```train.py``` as per your requirements.

## Citation
Preprint Citation
```
@article{brateanu2024,
  title={LYT-Net: Lightweight YUV Transformer-based Network for Low-Light Image Enhancement},
  author={Brateanu, Alexandru and Balmez, Raul and Avram, Adrian and Orhei, Ciprian},
  journal={arXiv preprint arXiv:2401.15204},
  year={2024}
}
```
