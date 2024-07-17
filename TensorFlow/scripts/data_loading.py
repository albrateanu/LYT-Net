import glob
import os
import tensorflow as tf


tf.random.set_seed(
    100
    )


def data_augmentation(raw_img, corrected_img):
    tf.random.set_seed(
    100
    )
    flip_lr = tf.random.uniform(shape=[]) > 0.5
    flip_ud = tf.random.uniform(shape=[]) > 0.5
    rot_k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)

    if flip_lr:
        raw_img = tf.image.flip_left_right(raw_img)
        corrected_img = tf.image.flip_left_right(corrected_img)

    if flip_ud:
        raw_img = tf.image.flip_up_down(raw_img)
        corrected_img = tf.image.flip_up_down(corrected_img)

    raw_img = tf.image.rot90(raw_img, k=rot_k)
    corrected_img = tf.image.rot90(corrected_img, k=rot_k)

    return raw_img, corrected_img


def load_image_test(image_path, crop_margin):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    
    original_shape = tf.shape(img)
    new_height = original_shape[0] - 2 * crop_margin
    new_width = original_shape[1] - 2 * crop_margin

    img = tf.image.crop_to_bounding_box(img, crop_margin, crop_margin, new_height, new_width)
    img = (tf.cast(img, tf.float32) / 127.5) - 1.0
    return img


def load_and_preprocess_image(raw_img_path, corrected_img_path):
    tf.random.set_seed(
    100
    )

    raw_img = tf.io.read_file(raw_img_path)
    raw_img = tf.image.decode_png(raw_img, channels=3)

    corrected_img = tf.io.read_file(corrected_img_path)
    corrected_img = tf.image.decode_png(corrected_img, channels=3)

    raw_img = tf.cast(raw_img, tf.float32)
    corrected_img = tf.cast(corrected_img, tf.float32)
    
    stacked_images = tf.stack([raw_img, corrected_img], axis=0) 
    cropped_images = tf.image.random_crop(stacked_images, size=[2, 256, 256, 3]) 
    raw_img, corrected_img = cropped_images[0], cropped_images[1] 

    raw_img = (raw_img / 255.0) * 2 - 1.0
    corrected_img = (corrected_img / 255.0) * 2 - 1.0

    return raw_img, corrected_img

def get_datasets(raw_image_path, corrected_image_path):
    tf.random.set_seed(
    100
    )
    
    train_raw_files = sorted(glob.glob(raw_image_path))
    train_corrected_files = sorted(glob.glob(corrected_image_path))

    train_dataset = tf.data.Dataset.from_tensor_slices((train_raw_files, train_corrected_files))
    train_dataset = train_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    BATCH_SIZE = 1

    train_dataset = train_dataset.shuffle(buffer_size=1000).batch(BATCH_SIZE)

    return train_dataset
    

def get_datasets_metrics(raw_image_path, corrected_image_path, crop_margin):
    raw_image_files = sorted(glob.glob(raw_image_path))
    corrected_image_files = sorted(glob.glob(corrected_image_path))

    train_raw_files = raw_image_files
    train_corrected_files = corrected_image_files

    train_raw_dataset = tf.data.Dataset.from_tensor_slices(train_raw_files).map(lambda x: load_image_test(x, crop_margin), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_corrected_dataset = tf.data.Dataset.from_tensor_slices(train_corrected_files).map(lambda x: load_image_test(x, crop_margin), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train_dataset = tf.data.Dataset.zip((train_raw_dataset, train_corrected_dataset))

    BATCH_SIZE = 1
    train_dataset = train_dataset.batch(BATCH_SIZE)

    return train_dataset