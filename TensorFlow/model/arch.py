import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
import keras


class MSEFBlock(layers.Layer):
    def __init__(self, filters, **kwargs):
        super(MSEFBlock, self).__init__(**kwargs)
        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)
        self.depthwise_conv = layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same')
        self.se_attn = SEBlock(filters)
        
    def call(self, inputs):
        x = self.layer_norm(inputs)
        x1 = self.depthwise_conv(x)
        x2 = self.se_attn(x)
        x_fused = layers.Multiply()([x1, x2])
        x_out = layers.Add()([x_fused, inputs])
        return x_out
    
class SEBlock(layers.Layer):
    def __init__(self, input_channels, reduction_ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.pool = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(input_channels // reduction_ratio, activation='relu')
        self.fc2 = layers.Dense(input_channels, activation='tanh')
    def call(self, inputs):
        x = self.pool(inputs)
        x = self.fc1(x)
        x = self.fc2(x)
        scale = tf.reshape(x, [-1, 1, 1, inputs.shape[-1]])
        return inputs * scale

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        assert embed_size % num_heads == 0, "Embedding size must be divisible by number of heads"
        self.head_dim = embed_size // num_heads
        self.query_dense = layers.Dense(embed_size)
        self.key_dense = layers.Dense(embed_size)
        self.value_dense = layers.Dense(embed_size)
        self.combine_heads = layers.Dense(embed_size)
        
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    def attention(self, query, key, value):
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        depth = tf.cast(tf.shape(key)[-1], tf.float32)
        logits = matmul_qk / tf.math.sqrt(depth)
        attention_weights = tf.nn.softmax(logits, axis=-1)
        output = tf.matmul(attention_weights, value)
        return output
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        
        attention = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_size))
        output = self.combine_heads(concat_attention)
        output = tf.reshape(output, [batch_size, height, width, self.embed_size])
        return output

class Denoiser(Model):
    def __init__(self, num_filters, kernel_size=3, activation='relu'):
        super(Denoiser, self).__init__()
        self.conv1 = layers.Conv2D(num_filters, kernel_size=kernel_size, strides=1, padding='same', activation=activation)
        self.conv2 = layers.Conv2D(num_filters, kernel_size=kernel_size, strides=2, padding='same', activation=activation)
        self.conv3 = layers.Conv2D(num_filters, kernel_size=kernel_size, strides=2, padding='same', activation=activation)
        self.conv4 = layers.Conv2D(num_filters, kernel_size=kernel_size, strides=2, padding='same', activation=activation)
        self.bottleneck = MultiHeadSelfAttention(embed_size=num_filters, num_heads=4)
        self.up2 = layers.UpSampling2D(2)
        self.up3 = layers.UpSampling2D(2)
        self.up4 = layers.UpSampling2D(2)
        self.output_layer = layers.Conv2D(1, kernel_size=kernel_size, strides=1, padding='same', activation='tanh')
        self.res_layer = layers.Conv2D(1, kernel_size=kernel_size, strides=1, padding='same', activation='tanh')
    def call(self, inputs):
        x1 = self.conv1(inputs)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x = self.bottleneck(x4)
        x = self.up4(x)
        x = self.up3(x3 + x)
        x = self.up2(x2 + x)
        x = x + x1
        x = self.res_layer(x)
        return self.output_layer(x + inputs)

class LYT(Model):
    def __init__(self, filters=32, denoiser_cb=None, denoiser_cr=None):
        super(LYT, self).__init__()
        self.process_y = self._create_processing_layers(filters)
        self.process_cb = self._create_processing_layers(filters)
        self.process_cr = self._create_processing_layers(filters)
       
        self.denoiser_cb = denoiser_cb
        
        self.denoiser_cr = denoiser_cr
        self.lum_pool = layers.MaxPooling2D(8)
        self.lum_mhsa = MultiHeadSelfAttention(embed_size=filters, num_heads=4)
        self.lum_up = layers.UpSampling2D(8)
        self.lum_conv = layers.Conv2D(filters, (1,1), padding='same')
        self.ref_conv = layers.Conv2D(filters, (1,1), padding='same')
        self.msef = MSEFBlock(filters)
        
        self.recombine = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')
        
        self.final_adjustments = layers.Conv2D(3, (3, 3), activation='tanh', padding='same')
    def _create_processing_layers(self, filters):
        layerz = [layers.Conv2D(filters, (3, 3), activation='relu', padding='same') for _ in range(1)]
        return keras.Sequential(layerz)
        
    def call(self, inputs):
        ycbcr = tf.image.rgb_to_yuv(inputs)
        y, cb, cr = tf.split(ycbcr, 3, axis=-1)
        cb = self.denoiser_cb(cb) + cb
        cr = self.denoiser_cr(cr) + cr
        
        y_processed = self.process_y(y)
        cb_processed = self.process_cb(cb)
        cr_processed = self.process_cr(cr)
        
        ref = tf.concat([cb_processed, cr_processed], axis=-1)
        
        lum = y_processed
        lum_1 = self.lum_pool(lum)
        lum_1 = self.lum_mhsa(lum_1)
        lum_1 = self.lum_up(lum_1)
        lum = lum + lum_1
        ref = self.ref_conv(ref)
        shortcut = ref
        ref = ref + 0.2 * self.lum_conv(lum)
        ref = self.msef(ref)
        ref = ref + shortcut
        
        recombined = self.recombine(tf.concat([ref, lum], axis=-1))
        output = self.final_adjustments(recombined)
        return output
