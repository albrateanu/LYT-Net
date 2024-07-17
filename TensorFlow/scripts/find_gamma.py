import tensorflow as tf
import numpy as np

def adjust_gamma(image, gamma=1.0):
    max_val = tf.reduce_max(image)
    image_normalized = image / max_val
    image_gamma_corrected = tf.pow(image_normalized, gamma)
    image_gamma_corrected = image_gamma_corrected * max_val
    return image_gamma_corrected

def find_optimal_gamma(source_image, target_image, gamma_range, max_pixel_value):
    optimal_gamma = gamma_range[0]
    max_psnr = 0
    for gamma in gamma_range:
        adjusted_image = adjust_gamma(source_image, gamma)
        psnr = tf.image.psnr(adjusted_image, target_image, max_pixel_value)
        if psnr > max_psnr:
            max_psnr = psnr
            optimal_gamma = gamma
    return optimal_gamma