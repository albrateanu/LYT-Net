import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(script_dir, '..')
sys.path.append(root_dir)

import data_loading as dl
import tensorflow as tf
from model.arch import LYT, Denoiser
from model.losses import load_vgg, loss
from model.scheduler import CosineDecayWithRestartsLearningRateSchedule
import argparse
import datetime
import numpy as np
from find_gamma import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(
    1
)

def get_time():
    current_time = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    return current_time

@tf.function
def train_step(raw_images, corrected_images, model, loss_model, optimizer):
    with tf.GradientTape() as tape:
        generated_images = model(raw_images, training=True)
        loss_val = loss(generated_images, corrected_images, loss_model)

    gradients = tape.gradient(loss_val, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    del tape
    return loss_val

def train(model, dataset, loss_model, optimizer, train_dataset, test_dataset, epochs, start_epoch=1):
    best_psnr = 0
    best_gt_psnr = 0
    best_ssim = 0
    num_samples = 0
    os.makedirs(f'./experiments/{dataset}', exist_ok=True)

    for epoch in range(start_epoch, epochs+1):
        for raw_images, corrected_images in train_dataset:
            loss = train_step(raw_images, corrected_images, model, loss_model, optimizer)
            
        total_psnr = 0
        total_ssim = 0
        total_gt_psnr = 0
        num_samples = 0
        for raw_images, corrected_images in test_dataset:
                generated_images = model(raw_images)
                generated_images = (generated_images + 1.0) / 2.0
                corrected_images = (corrected_images + 1.0) / 2.0
                psnr = tf.image.psnr(corrected_images, generated_images, max_val=1.0)
                gamma_values = np.linspace(0.4, 2.5, 100)
                optimal_gamma = find_optimal_gamma(generated_images, corrected_images, gamma_values, 1.0)
                generated_images = adjust_gamma(generated_images, optimal_gamma)
                gt_psnr = tf.image.psnr(corrected_images, generated_images, max_val=1.0)
                ssim = tf.image.ssim(corrected_images, generated_images, max_val=1.0)

                total_psnr += tf.reduce_mean(psnr)
                total_ssim += tf.reduce_mean(ssim)
                total_gt_psnr += tf.reduce_mean(gt_psnr)
                num_samples += 1
        avg_psnr = total_psnr / num_samples
        avg_ssim = total_ssim / num_samples
        avg_gt_psnr = total_gt_psnr / num_samples
        print(f"({get_time()}) Epoch {epoch} | GT-PSNR: {avg_gt_psnr:.2f} | PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.3f} | loss={loss:.6f}")
        if avg_gt_psnr > best_gt_psnr:
            best_psnr, best_ssim = avg_psnr, avg_ssim
            best_gt_psnr = avg_gt_psnr
            model_name = f"net_psnr_{best_gt_psnr:.2f}_ssim_{best_ssim:.3f}_epoch_{epoch}_dataset_{dataset}.h5"
            model.save_weights(os.path.join(f'./experiments/{dataset}', model_name))
            print(f"({get_time()}) Saved model as {model_name}.")


def start_train(dataset):
    print('LYT-Net 2024 (c) Brateanu, A., Balmez, R., Avram A., Orhei, C.C.')
    print(f"({get_time()}) Training on dataset: {dataset}")

    raw_image_path = ''
    corrected_image_path = ''
    raw_test_path = ''
    corrected_test_path = ''

    # Load dataset
    if dataset == 'LOLv1':
        raw_image_path = './data/LOLv1/Train/input/*.png'
        corrected_image_path = './data/LOLv1/Train/target/*.png'

        raw_test_path = './data/LOLv1/Test/input/*.png'
        corrected_test_path = './data/LOLv1/Test/target/*.png'

    elif dataset == 'LOLv2_Real':
        raw_image_path = './data/LOLv2/Real_captured/Train/Low/*.png'
        corrected_image_path = './data/LOLv2/Real_captured/Train/Normal/*.png'

        raw_test_path = './data/LOLv2/Real_captured/Test/Low/*.png'
        corrected_test_path = './data/LOLv2/Real_captured/Test/Normal/*.png'

    elif dataset == 'LOLv2_Synthetic':
        raw_image_path = './data/LOLv2/Synthetic/Train/Low/*.png'
        corrected_image_path = './data/LOLv2/Synthetic/Train/Normal/*.png'

        raw_test_path = './data/LOLv2/Synthetic/Test/Low/*.png'
        corrected_test_path = './data/LOLv2/Synthetic/Test/Normal/*.png'

    else:
        print('Incorrect usage. \'--dataset\' argument must be one of [LOLv1/LOLv2_Real/LOLv2_Synthetic].')
        return

    print(f'({get_time()}) Loading dataset {dataset}.')
    train_dataset = dl.get_datasets(raw_image_path, corrected_image_path)
    test_dataset = dl.get_datasets_metrics(raw_test_path, corrected_test_path, 0)
    print(f'({get_time()}) Successfully loaded dataset.')
    
    # Build model
    denoiser_cb = Denoiser(16)
    denoiser_cr = Denoiser(16)
    denoiser_cb.build(input_shape=(None,None,None,1))
    denoiser_cr.build(input_shape=(None,None,None,1))
    model = LYT(filters=32, denoiser_cb=denoiser_cb, denoiser_cr=denoiser_cr)
    model.build(input_shape=(None,None,None,3))
    # Loss Function Req
    loss_model = load_vgg()

    # Optimizer
    initial_lr = 2e-4
    min_lr = 1e-6
    total_epochs = 1000
    steps_per_epoch = len(train_dataset)
    total_steps = total_epochs * steps_per_epoch
    first_decay_steps = 150*len(train_dataset)

    cosine_ann_lr_fn = CosineDecayWithRestartsLearningRateSchedule(
        initial_lr, min_lr, total_steps, first_decay_steps
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=cosine_ann_lr_fn)

    # Run training
    print(f'({get_time()}) Starting training.')
    train(model=model,
          dataset=dataset,
          loss_model=loss_model,
          optimizer=optimizer,
          train_dataset=train_dataset,
          test_dataset=test_dataset,
          epochs=1000)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name for training')
    args = parser.parse_args()

    start_train(args.dataset)

