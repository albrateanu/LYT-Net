import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(script_dir, '..')
sys.path.append(root_dir)

import data_loading as dl
import tensorflow as tf
from model.arch import LYT, Denoiser
import argparse
import datetime
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(
    1
)

def get_time():
    current_time = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    return current_time

def start_test(dataset):
    print('LYT-Net 2024 (c) Brateanu, A., Balmez, R., Avram A., Orhei, C.C.')
    print(f"({get_time()}) Testing on dataset: {dataset}")

    raw_test_path = ''
    corrected_test_path = ''
    model_name = ''

    # Load dataset
    if dataset == 'LOLv1':
        raw_test_path = './data/LOLv1/Test/input/*.png'
        corrected_test_path = './data/LOLv1/Test/target/*.png'
        model_name = 'LOL_v1_weights.h5'

    elif dataset == 'LOLv2_Real':
        raw_test_path = './data/LOLv2/Real_captured/Test/Low/*.png'
        corrected_test_path = './data/LOLv2/Real_captured/Test/Normal/*.png'
        model_name = 'LOL_v2_real_weights.h5'

    elif dataset == 'LOLv2_Synthetic':
        raw_test_path = './data/LOLv2/Synthetic/Test/Low/*.png'
        corrected_test_path = './data/LOLv2/Synthetic/Test/Normal/*.png'
        model_name = 'LOL_v2_synthetic_weights.h5'

    else:
        print('Incorrect usage. \'--dataset\' argument must be one of [LOLv1/LOLv2_Real/LOLv2_Synthetic].')
        return
    
    # Load dataset
    print(f'({get_time()}) Loading dataset {dataset}.')
    test_dataset = dl.get_datasets_metrics(raw_test_path, corrected_test_path, 0)
    print(f'({get_time()}) Successfully loaded dataset.')

    # Build model
    denoiser_cb = Denoiser(16)
    denoiser_cr = Denoiser(16)
    denoiser_cb.build(input_shape=(None,None,None,1))
    denoiser_cr.build(input_shape=(None,None,None,1))
    model = LYT(filters=32, denoiser_cb=denoiser_cb, denoiser_cr=denoiser_cr)
    model.build(input_shape=(None,None,None,3))

    # Loading weights
    model.load_weights(f'./pretrained_weights/{model_name}')


    total_psnr = 0
    total_ssim = 0
    num_samples = 0

    for raw_image, corrected_image in tqdm(test_dataset):
            generated_image = model(raw_image)
            raw_image = (raw_image + 1.0) / 2.0
            corrected_image = (corrected_image + 1.0) / 2.0
            generated_image = (generated_image + 1.0) / 2.0
        
            total_psnr += tf.image.psnr(corrected_image, generated_image, max_val=1.0)
            total_ssim += tf.image.ssim(corrected_image, generated_image, max_val=1.0)
        
            num_samples += 1
        
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples

    avg_psnr = tf.reduce_mean(avg_psnr)
    avg_ssim = tf.reduce_mean(avg_ssim)

    print(f"PSNR: {avg_psnr:.6f}")
    print(f"SSIM: {avg_ssim:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing script')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name for testing')
    args = parser.parse_args()

    start_test(args.dataset)
