import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras import Model


def color_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(tf.reduce_mean(y_true, axis=[1, 2]) - tf.reduce_mean(y_pred, axis=[1, 2])))

def psnr_loss(y_true, y_pred):
    return 40.0-tf.image.psnr(y_true, y_pred, max_val=1.0)

def load_vgg():
    vgg = VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    return loss_model

def perceptual_loss(y_true, y_pred, loss_model):
    return tf.reduce_mean(tf.square(loss_model(y_true) - loss_model(y_pred)))

def smooth_l1_loss(y_true, y_pred):
    diff = tf.abs(y_true - y_pred)
    less_than_one = tf.cast(tf.less(diff, 1.0), tf.float32)
    smooth_l1_loss = (0.5 * diff**2) * less_than_one + (diff - 0.5) * (1.0 - less_than_one)
    return tf.reduce_mean(smooth_l1_loss)

def multiscale_ssim_loss(y_true, y_pred, max_val=1.0, power_factors=[0.5, 0.5]):
    return 1 - tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, max_val, power_factors=power_factors))

def histogram_loss(y_true, y_pred, bins=256, sigma=0.01):
    bin_edges = tf.linspace(0.0, 1.0, bins)
    
    def gaussian_kernel(x, mu, sigma):
        return tf.exp(-0.5 * ((x - mu) / sigma) ** 2)

    y_true_hist = tf.reduce_sum(gaussian_kernel(y_true[..., tf.newaxis], bin_edges, sigma), axis=0)
    y_pred_hist = tf.reduce_sum(gaussian_kernel(y_pred[..., tf.newaxis], bin_edges, sigma), axis=0)
    
    y_true_hist /= tf.reduce_sum(y_true_hist)
    y_pred_hist /= tf.reduce_sum(y_pred_hist)

    hist_distance = tf.reduce_mean(tf.abs(y_true_hist - y_pred_hist))

    return hist_distance

def loss(y_true, y_pred, loss_model):
    y_true = (y_true + 1.0) / 2.0
    y_pred = (y_pred + 1.0) / 2.0
    alpha1 = 1.00
    alpha2 = 0.06
    alpha3 = 0.05
    alpha4 = 0.5
    alpha5 = 0.0083
    alpha6 = 0.25

    smooth_l1_l = smooth_l1_loss(y_true, y_pred)
    ms_ssim_l = multiscale_ssim_loss(y_true, y_pred)
    perc_l = perceptual_loss(y_true, y_pred, loss_model=loss_model)
    hist_l = histogram_loss(y_true, y_pred)
    psnr_l = psnr_loss(y_true, y_pred)
    color_l = color_loss(y_true, y_pred)

    total_loss = alpha1 * smooth_l1_l + alpha2 * perc_l + alpha3*hist_l + alpha5*psnr_l + alpha6*color_l+ alpha4*ms_ssim_l
    total_loss = tf.reduce_mean(total_loss)
    return total_loss
