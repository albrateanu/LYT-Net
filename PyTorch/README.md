# LYT-Net: Lightweight YUV Transformer-based Network for Low-Light Image Enhancement

<div align="center">
  
<img src="figs/Logo.png" alt="LYT-Net Logo" width="300" height="300" />

[![arXiv](https://img.shields.io/badge/arxiv-paper-179bd3)](https://arxiv.org/abs/2401.15204)

## Description
This is the PyTorch version of LYT-Net.

## Experiment

### 1. Create Environment
- Make Conda Environment
```bash

```
- Install Dependencies
```bash

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
You can test the model using the following commands. Pre-trained weights are available at [Google Drive](). GT Mean evaluation can be done with the ```--gtmean``` argument.

```bash
# Test on LOLv1
# Test on LOLv1 using GT Mean

# Test on LOLv2 Real
# Test on LOLv2 Real using GT Mean

# Test on LOLv2 Synthetic
# Test on LOLv2 Synthetic using GT Mean
```

### 4. Compute Complexity
You can test the model complexity (FLOPS/Params) using the following command:
```bash
# To run FLOPS check with default (1,256,256,3)
python macs.py
```

### 5. Train
You can train the model using the following commands:

```bash
# Train on LOLv1

# Train on LOLv2 Real

# Train on LOLv2 Synthetic
```

**Note:** Please make sure to download VGG19 pre-trained weights for the hybrid loss function from here: [Google Drive](). Then, place it under the current directory (i.e. ```PyTorch```).

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
