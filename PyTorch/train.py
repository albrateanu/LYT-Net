import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
from torchvision.utils import save_image
from model import LYT
from losses import CombinedLoss
from dataloader import create_dataloaders
import os
import numpy as np

def _calculate_psnr(img1, img2, crop_border=0, input_order='CHW', test_y_channel=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio) with brightness normalization.

    Args:
        img1 (Tensor/ndarray): Prediction image with values in [0, 1] or [0, 255].
        img2 (Tensor/ndarray): Ground truth image with values in [0, 1] or [0, 255].
        crop_border (int): Number of pixels to crop from each border.
        input_order (str): 'HWC' or 'CHW'.
        test_y_channel (bool): Calculate PSNR on Y channel in YCbCr color space.

    Returns:
        float: PSNR value.
    """
    assert img1.shape == img2.shape, f'Image shapes are different: {img1.shape}, {img2.shape}.'
    
    if isinstance(img1, torch.Tensor):
        img1 = img1.permute(1, 2, 0).detach().cpu().numpy() if input_order == 'CHW' else img1.detach().cpu().numpy()
        img2 = img2.permute(1, 2, 0).detach().cpu().numpy() if input_order == 'CHW' else img2.detach().cpu().numpy()

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border]

    # # Convert images to grayscale by averaging channels
    if img1.ndim == 3:
        img1_gray = img1.mean(axis=2)
        img2_gray = img2.mean(axis=2)
    else:
        img1_gray = img1
        img2_gray = img2

    # Normalize brightness
    mean_restored = img1_gray.mean()
    mean_target = img2_gray.mean()
    img1 = np.clip(img1 * (mean_target / mean_restored), 0, 1)

    # Calculate MSE on normalized image
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')

    max_value = 1.0 if img1.max() <= 1 else 255.0
    psnr = 20 * np.log10(max_value / np.sqrt(mse))

    return psnr

def calculate_psnr(img1, img2, max_pixel_value=1.0):
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio) between two images.

    Args:
        img1 (torch.Tensor): First image (BxCxHxW)
        img2 (torch.Tensor): Second image (BxCxHxW)
        max_pixel_value (float): The maximum possible pixel value of the images. Default is 1.0.

    Returns:
        float: The PSNR value.
    """
    img1_gray = img1.mean(axis=1)
    img2_gray = img2.mean(axis=1)
    
    mean_restored = img1_gray.mean()
    mean_target = img2_gray.mean()
    img1 = np.clip(img1 * (mean_target / mean_restored), 0, 1)
    
    mse = F.mse_loss(img1, img2, reduction='mean')
    # print(f'MSE: {mse.item()}')
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    # print(f'PSNR: {psnr.item()}')
    return psnr.item()

def validate(model, dataloader, device):
    model.eval()
    total_psnr = 0
    total_ssim = 0
    with torch.no_grad():
        for low, high in dataloader:
            low, high = low.to(device), high.to(device)
            output = model(low)

            # Calculate PSNR
            psnr = calculate_psnr(output, high)
            total_psnr += psnr


    avg_psnr = total_psnr / len(dataloader)
    avg_ssim = total_ssim / len(dataloader)
    return avg_psnr, avg_ssim

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

def main():
    # Hyperparameters
    train_low = 'data/LOLv1/Train/input'
    train_high = 'data/LOLv1/Train/target'
    test_low = 'data/LOLv1/Test/input'
    test_high = 'data/LOLv1/Test/target'
    learning_rate = 2e-4 
    num_epochs = 1500
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'LR: {learning_rate}; Epochs: {num_epochs}')

    # Data loaders
    train_loader, test_loader = create_dataloaders(train_low, train_high, test_low, test_high)
    print(f'Train loader: {len(train_loader)}; Test loader: {len(test_loader)}')

    # Model, loss, optimizer, and scheduler
    model = LYT().to(device)
    model.apply(weights_init)
    criterion = CombinedLoss(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_psnr = 0
    print('Training started.')
    for epoch in range(num_epochs):
        model.train()
        for low, high in train_loader:
            low, high = low.to(device), high.to(device)
            optimizer.zero_grad()
            output = model(low)
            loss = criterion(output, high)
            # print(f'Loss: {loss.item()}')  # Print the loss value
            loss.backward()
            optimizer.step()

        avg_psnr, avg_ssim = validate(model, test_loader, device)
        print(f'Epoch {epoch + 1}/{num_epochs}, PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}')
        scheduler.step()

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Saving model with PSNR: {best_psnr:.2f}')

if __name__ == '__main__':
    main()
