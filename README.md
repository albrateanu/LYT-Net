# LYT-Net: Lightweight YUV Transformer-based Network for Low-Light Image Enhancement

<div align="center">
  
[![arXiv](https://img.shields.io/badge/arxiv-paper-179bd3)](https://arxiv.org/abs/2401.15204)
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/lyt-net-lightweight-yuv-transformer-based/low-light-image-enhancement-on-lol)](https://paperswithcode.com/sota/low-light-image-enhancement-on-lol?p=lyt-net-lightweight-yuv-transformer-based)
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/lyt-net-lightweight-yuv-transformer-based/low-light-image-enhancement-on-lolv2)](https://paperswithcode.com/sota/low-light-image-enhancement-on-lolv2?p=lyt-net-lightweight-yuv-transformer-based)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/lyt-net-lightweight-yuv-transformer-based/low-light-image-enhancement-on-lolv2-1)](https://paperswithcode.com/sota/low-light-image-enhancement-on-lolv2-1?p=lyt-net-lightweight-yuv-transformer-based)

Ranked #1 on FLOPS(G) (3.49 GFLOPS) and Params(M) (0.045M = 45k Params)
</div>

## Updates
<!-- - `12.01.2024`: text update -->
- `03.04.2024` Training code re-added and adjusted.
- `30.01.2024` arXiv pre-print available.
- `10.01.2024` Pre-trained model weights and code for training and testing are released.

## Experiment

### 1. Create Environment
- Make Conda Environment
```bash
conda create -n LYTNet python=3.10
conda activate LYTNet
```
- Install Dependencies
```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1
pip install tensorflow==2.10 opencv-python numpy tqdm matplotlib lpips
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

### 3. Test
You can test the model using the following commands. Pre-trained weights are available at [Google Drive](https://drive.google.com/drive/folders/1LgLUXGy-7fQXVnxyEeyBolkZ5ZX1f_em?usp=sharing). GT Mean evaluation can be done with the ```--gtmean``` argument.

```bash
# Test on LOLv1
python main.py --test --dataset LOLv1 --weights pretrained_weights/LOL_v1.h5
# Test on LOLv1 using GT Mean
python main.py --test --dataset LOLv1 --weights pretrained_weights/LOL_v1.h5 --gtmean

# Test on LOLv2 Real
python main.py --test --dataset LOLv2_Real --weights pretrained_weights/LOL_v2_real.h5
# Test on LOLv2 Real using GT Mean
python main.py --test --dataset LOLv2_Real --weights pretrained_weights/LOL_v2_real.h5 --gtmean

# Test on LOLv2 Synthetic
python main.py --test --dataset LOLv2_Synthetic --weights pretrained_weights/LOL_v2_synthetic.h5
# Test on LOLv2 Synthetic using GT Mean
python main.py --test --dataset LOLv2_Synthetic --weights pretrained_weights/LOL_v2_synthetic.h5 --gtmean
```

### 4. Compute Complexity
You can test the model complexity (FLOPS/Params) using the following command:
```bash
# To run FLOPS check with default (1,256,256,3)
python main.py --complexity

# To run FLOPS check with custom (1,H,W,C)
python main.py --complexity --shape '(H,W,C)'
```

### 5. Train
You can train the model using the following commands:

```bash
# Train on LOLv1
python main.py --train --dataset LOLv1

# Train on LOLv2 Real
python main.py --train --dataset LOLv2_Real

# Train on LOLv2 Synthetic
python main.py --train --dataset LOLv2_Synthetic
```

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
