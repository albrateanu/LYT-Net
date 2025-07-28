# [SPL 2025] LYT-Net: Lightweight YUV Transformer-based Network for Low-Light Image Enhancement

<div align="center">

![Logo](./figs/Logo.png)

**[Alexandru Brateanu](https://scholar.google.com/citations?user=ru0meGgAAAAJ&hl=en), [Raul Balmez](https://scholar.google.com/citations?user=vPC7raQAAAAJ&hl=en), [Adrian Avram](https://scholar.google.com/citations?user=Wk3IxkEAAAAJ&hl=en), [Ciprian Orhei](https://scholar.google.com/citations?user=DZHdq3wAAAAJ&hl=en), [Cosmin Ancuti](https://scholar.google.com/citations?user=zVTgt8IAAAAJ&hl=en)**


**Check out our HuggingFace page for LYT-Net!**

[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-179bd3)](https://huggingface.co/albrateanu/LYT-Net)

[![arXiv](https://img.shields.io/badge/arxiv-paper-179bd3)](https://arxiv.org/abs/2401.15204)
[![IEEE](https://img.shields.io/badge/IEEE-paper-blue)](https://ieeexplore.ieee.org/abstract/document/10972228)
	
<!---[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/lyt-net-lightweight-yuv-transformer-based/low-light-image-enhancement-on-lol)](https://paperswithcode.com/sota/low-light-image-enhancement-on-lol?p=lyt-net-lightweight-yuv-transformer-based)
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/lyt-net-lightweight-yuv-transformer-based/low-light-image-enhancement-on-lolv2)](https://paperswithcode.com/sota/low-light-image-enhancement-on-lolv2?p=lyt-net-lightweight-yuv-transformer-based)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/lyt-net-lightweight-yuv-transformer-based/low-light-image-enhancement-on-lolv2-1)](https://paperswithcode.com/sota/low-light-image-enhancement-on-lolv2-1?p=lyt-net-lightweight-yuv-transformer-based) -->



Ranked #1 on FLOPS(G) (3.49 GFLOPS) and Params(M) (0.045M = 45k Params)
</div>

### Abstract

*This letter introduces LYT-Net, a novel lightweight transformer-based model for low-light image enhancement (LLIE). LYT-Net consists of several layers and detachable blocks, including our novel blocks--Channel-Wise Denoiser (CWD) and Multi-Stage Squeeze & Excite Fusion (MSEF)--along with the traditional Transformer block, Multi-Headed Self-Attention (MHSA). In our method we adopt a dual-path approach, treating chrominance channels U and V and luminance channel Y as separate entities to help the model better handle illumination adjustment and corruption restoration. Our comprehensive evaluation on established LLIE datasets demonstrates that, despite its low complexity, our model outperforms recent LLIE methods. The source code and pre-trained models are available at [this https URL](https://github.com/albrateanu/LYT-Net).*


## 🆕 Updates
- `28.07.2025` ✨ Check out our new multimodal framework for LLIE: **ModalFormer: Multimodal Transformer for Low-Light Image Enhancement**! Paper and HF Demo coming soon!
- `27.07.2025` 🤗 LYT-Net now has a new HuggingFace page! Check it out [here](https://huggingface.co/albrateanu/LYT-Net)! **HF Demo coming soon!** 
- `09.05.2025` 📢 Check out our other works on [Low-light Image Enhancement](https://github.com/albrateanu/KANT) and [Image Denoising](https://github.com/albrateanu/AKDT)!
- `21.04.2025` 📝 LYT-Net is published as a IEEE Signal Processing Letters paper. [Link to paper](https://ieeexplore.ieee.org/abstract/document/10972228).
- `17.07.2024` 🧪 Released rudimentary PyTorch implementation.
- `03.04.2024` 🔧 Training code re-added and adjusted.
- `30.01.2024` 📄 arXiv pre-print available.
- `10.01.2024` 🚀 Pre-trained model weights and code for training and testing are released.

## 🧪 Experiment
Please check the ```TensorFlow``` and ```PyTorch``` folders for library-specific implementations.

## 📊 Results

| Dataset  | TensorFlow |           | PyTorch |           |
|:--------:|:----------:|:---------:|:-------:|:---------:|
|          | PSNR       | SSIM      | PSNR    | SSIM      |
|  LOLv1   |  27.23     |  0.853    | 26.63   |  0.836    |
| LOLv2-R  |  27.80     |  0.873    | 28.41   |  0.878    |
| LOLv2-S  |  29.39     |  0.939    | 26.72   |  0.928    |


## 📚 Citation
```
@article{brateanu2025lyt,
  author={Brateanu, Alexandru and Balmez, Raul and Avram, Adrian and Orhei, Ciprian and Ancuti, Cosmin},
  journal={IEEE Signal Processing Letters}, 
  title={LYT-NET: Lightweight YUV Transformer-based Network for Low-light Image Enhancement}, 
  year={2025},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/LSP.2025.3563125}}


@article{brateanu2024lyt,
  title={LYT-Net: Lightweight YUV Transformer-based Network for Low-Light Image Enhancement},
  author={Brateanu, Alexandru and Balmez, Raul and Avram, Adrian and Orhei, Ciprian and Cosmin, Ancuti},
  journal={arXiv preprint arXiv:2401.15204},
  year={2024}
}
```
