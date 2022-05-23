import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

import math


#########################
# ## SNGAN-BASED EBM ## #
#########################

# ebm architectures derived from SNGAN architectures, with spectral normalization removed throughout.

# TF2 Keras reimplementation of Pytorch SN-GAN code from Mimicry Git Repo 
# https://github.com/kwotsin/mimicry
# Original Code: Copyright (c) 2020 Kwot Sin Lee under MIT License

class EBMBlock(keras.layers.Layer):
  def __init__(self, in_channels, out_channels, hidden_channels=None, downsample=False):
    super().__init__()

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels if hidden_channels is not None else out_channels
    self.learnable_sc = in_channels != out_channels or downsample
    self.downsample = downsample

    self.conv_1 = keras.layers.Conv2D(filters=self.hidden_channels, kernel_size=3, padding="SAME")
    self.conv_2 = keras.layers.Conv2D(filters=out_channels, kernel_size=3, padding="SAME")

    self.relu = keras.activations.relu

    if self.learnable_sc:
      self.sc = keras.layers.Conv2D(filters=out_channels, kernel_size=1, padding="VALID")
    if self.downsample:
      self.downsampling_layer = keras.layers.AveragePooling2D()

  def call(self, x, training=False):

    # residual layer
    h = x
    h = self.relu(h)
    h = self.conv_1(h)
    h = self.relu(h)
    h = self.conv_2(h)
    h = self.downsampling_layer(h) if self.downsample else h

    # shortcut
    y = x
    y = self.sc(y) if self.learnable_sc else y
    y = self.downsampling_layer(y) if self.downsample else y

    return h + y

class EBMBlockStem(keras.layers.Layer):
  def __init__(self, in_channels, out_channels, hidden_channels=None, downsample=False):
    super().__init__()

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels if hidden_channels is not None else out_channels
    self.learnable_sc = in_channels != out_channels or downsample
    self.downsample = downsample

    self.conv_1 = keras.layers.Conv2D(filters=self.hidden_channels, kernel_size=3, padding="SAME")
    self.conv_2 = keras.layers.Conv2D(filters=out_channels, kernel_size=3, padding="SAME")

    self.relu = keras.activations.relu

    if self.learnable_sc:
      self.sc = keras.layers.Conv2D(filters=out_channels, kernel_size=1, padding="VALID")
    if self.downsample:
      self.downsampling_layer = keras.layers.AveragePooling2D()

  def call(self, x, training=False):

    # residual layer
    h = x
    h = self.conv_1(h)
    h = self.relu(h)
    h = self.conv_2(h)
    h = self.downsampling_layer(h) if self.downsample else h

    # shortcut
    y = x
    y = self.downsampling_layer(y) if self.downsample else y
    y = self.sc(y) if self.learnable_sc else y

    return h + y

class EBMSNGAN32(keras.Model):
  def __init__(self, ngf=256):
    super().__init__()

    self.ngf = ngf

    # Build the layers
    self.conv_1 = EBMBlockStem(3, self.ngf, downsample=True)
    self.block2 = EBMBlock(self.ngf, self.ngf, downsample=True)
    self.block3 = EBMBlock(self.ngf, self.ngf, downsample=False)
    self.block4 = EBMBlock(self.ngf, self.ngf, downsample=False)
    self.pool_5 = keras.layers.GlobalAveragePooling2D()
    self.lin_5 = keras.layers.Dense(1, use_bias=False)

  def call(self, x, training=False):

    x = self.conv_1(x)

    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)

    x = keras.activations.relu(x)
    x = self.pool_5(x)
    x = self.lin_5(x)

    return x

class EBMSNGAN64(keras.Model):
  def __init__(self, ngf=1024):
    super().__init__()

    self.ngf = ngf

    # Build the layers
    self.conv_1 = EBMBlockStem(3, self.ngf >> 4, downsample=True)
    self.block2 = EBMBlock(self.ngf >> 4, self.ngf >> 3, downsample=True)
    self.block3 = EBMBlock(self.ngf >> 3, self.ngf >> 2, downsample=True)
    self.block4 = EBMBlock(self.ngf >> 2, self.ngf >> 1, downsample=True)
    self.block5 = EBMBlock(self.ngf >> 1, self.ngf, downsample=True)
    self.pool_6 = keras.layers.GlobalAveragePooling2D()
    self.lin_6 = keras.layers.Dense(1, use_bias=False)

  def call(self, x, training=False):

    x = self.conv_1(x)

    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    x = self.block5(x)

    x = keras.activations.relu(x)
    x = self.pool_6(x)
    x = self.lin_6(x)

    return x

class EBMSNGAN128(keras.Model):
  def __init__(self, ngf=1024):
    super().__init__()

    self.ngf = ngf

    # Build the layers
    self.conv_1 = EBMBlockStem(3, self.ngf >> 4, downsample=True)
    self.block2 = EBMBlock(self.ngf >> 4, self.ngf >> 3, downsample=True)
    self.block3 = EBMBlock(self.ngf >> 3, self.ngf >> 2, downsample=True)
    self.block4 = EBMBlock(self.ngf >> 2, self.ngf >> 1, downsample=True)
    self.block5 = EBMBlock(self.ngf >> 1, self.ngf, downsample=True)
    self.block6 = EBMBlock(self.ngf, self.ngf, downsample=False)
    self.pool_7 = keras.layers.GlobalAveragePooling2D()
    self.lin_7 = keras.layers.Dense(1, use_bias=False)

  def call(self, x, training=False):

    x = self.conv_1(x)

    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    x = self.block5(x)
    x = self.block6(x)

    x = keras.activations.relu(x)
    x = self.pool_7(x)
    x = self.lin_7(x)

    return x


#########################
# ## SNGAN GENERATOR ## #
#########################

# TF2 Keras reimplementation of Pytorch SN-GAN code from Mimicry Git Repo 
# https://github.com/kwotsin/mimicry
# Original Code: Copyright (c) 2020 Kwot Sin Lee under MIT License

class GenBlock(keras.layers.Layer):
  def __init__(self, in_channels, out_channels, hidden_channels=None, upsample=False):
    super(GenBlock, self).__init__()

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels if hidden_channels is not None else out_channels
    self.learnable_sc = in_channels != out_channels or upsample
    self.upsample = upsample

    self.conv_1 = keras.layers.Conv2D(filters=self.hidden_channels, kernel_size=3, padding="SAME")
    self.conv_2 = keras.layers.Conv2D(filters=out_channels, kernel_size=3, padding="SAME")

    self.bn_1 = keras.layers.experimental.SyncBatchNormalization()
    self.bn_2 = keras.layers.experimental.SyncBatchNormalization()

    self.relu = keras.activations.relu

    if self.learnable_sc:
      self.sc = keras.layers.Conv2D(filters=out_channels, kernel_size=1, padding="VALID")
    if self.upsample:
      self.upsampling_layer = keras.layers.UpSampling2D(interpolation='bilinear')

  def call(self, x, training=False):

    # residual layer
    h = x
    h = self.bn_1(h, training=training)
    h = self.relu(h)
    h = self.upsampling_layer(h) if self.upsample else h
    h = self.conv_1(h)
    h = self.bn_2(h, training=training)
    h = self.relu(h)
    h = self.conv_2(h)

    # shortcut
    y = x
    y = self.upsampling_layer(y) if self.upsample else y
    y = self.sc(y) if self.learnable_sc else y

    return h + y

# 32x32 sngan generator
class GenSNGAN32(keras.Model):
  def __init__(self, nz=128, ngf=256, bottom_width=4):
    super().__init__()

    self.nz = nz
    self.ngf = ngf
    self.bottom_width = bottom_width

    # Build the layers
    self.lin_1 = keras.layers.Dense((self.bottom_width**2) * self.ngf)
    self.block2 = GenBlock(self.ngf, self.ngf, upsample=True)
    self.block3 = GenBlock(self.ngf, self.ngf, upsample=True)
    self.block4 = GenBlock(self.ngf, self.ngf, upsample=True)
    self.bn_5 = keras.layers.experimental.SyncBatchNormalization()
    self.relu = keras.activations.relu
    self.conv_5 = keras.layers.Conv2D(filters=3, kernel_size=3, padding="SAME")

  def generate_latent_z(self, num_ims):
    return tf.random.normal([num_ims, self.nz])  # noise sample

  def generate_images(self, num_ims):
    z = self.generate_latent_z(num_ims)
    return self.call(z)

  def call(self, x, training=True):

    h = self.lin_1(x)
    h = tf.reshape(h, (-1, self.ngf, self.bottom_width, self.bottom_width))
    h = tf.transpose(h, (0, 2, 3, 1))
    h = self.block2(h, training=training)
    h = self.block3(h, training=training)
    h = self.block4(h, training=training)
    h = self.bn_5(h, training=training)
    h = self.relu(h)
    h = self.conv_5(h)
    h = keras.activations.tanh(h)

    return h

# 64x64 sngan generator
class GenSNGAN64(keras.Model):
  def __init__(self, nz=128, ngf=1024, bottom_width=4):
    super().__init__()

    self.nz = nz
    self.ngf = ngf
    self.bottom_width = bottom_width

    # Build the layers
    self.lin_1 = keras.layers.Dense((self.bottom_width**2) * self.ngf)
    self.block2 = GenBlock(self.ngf, self.ngf >> 1, upsample=True)
    self.block3 = GenBlock(self.ngf >> 1, self.ngf >> 2, upsample=True)
    self.block4 = GenBlock(self.ngf >> 2, self.ngf >> 3, upsample=True)
    self.block5 = GenBlock(self.ngf >> 3, self.ngf >> 4, upsample=True)
    self.bn_6 = keras.layers.experimental.SyncBatchNormalization()
    self.relu = keras.activations.relu
    self.conv_6 = keras.layers.Conv2D(filters=3, kernel_size=3, padding="SAME")

  def generate_latent_z(self, num_ims):
    return tf.random.normal([num_ims, self.nz])  # noise sample

  def generate_images(self, num_ims):
    z = self.generate_latent_z(num_ims)
    return self.call(z)

  def call(self, x, training=True):

    h = self.lin_1(x)
    h = tf.reshape(h, (-1, self.ngf, self.bottom_width, self.bottom_width))
    h = tf.transpose(h, (0, 2, 3, 1))
    h = self.block2(h, training=training)
    h = self.block3(h, training=training)
    h = self.block4(h, training=training)
    h = self.block5(h, training=training)
    h = self.bn_6(h, training=training)
    h = self.relu(h)
    h = self.conv_6(h)
    h = keras.activations.tanh(h)

    return h

# 128x128 sngan generator
class GenSNGAN128(keras.Model):
  def __init__(self, nz=128, ngf=1024, bottom_width=4):
    super().__init__()

    self.nz = nz
    self.ngf = ngf
    self.bottom_width = bottom_width

    # Build the layers
    self.lin_1 = keras.layers.Dense((self.bottom_width**2) * self.ngf)
    self.block2 = GenBlock(self.ngf, self.ngf, upsample=True)
    self.block3 = GenBlock(self.ngf, self.ngf >> 1, upsample=True)
    self.block4 = GenBlock(self.ngf >> 1, self.ngf >> 2, upsample=True)
    self.block5 = GenBlock(self.ngf >> 2, self.ngf >> 3, upsample=True)
    self.block6 = GenBlock(self.ngf >> 3, self.ngf >> 4, upsample=True)
    self.bn_7 = keras.layers.experimental.SyncBatchNormalization()
    self.relu = keras.activations.relu
    self.conv_7 = keras.layers.Conv2D(filters=3, kernel_size=3, padding="SAME")

  def generate_latent_z(self, num_ims):
    return tf.random.normal([num_ims, self.nz])  # noise sample

  def generate_images(self, num_ims):
    z = self.generate_latent_z(num_ims)
    return self.call(z)

  def call(self, x, training=True):

    h = self.lin_1(x)
    h = tf.reshape(h, (-1, self.ngf, self.bottom_width, self.bottom_width))
    h = tf.transpose(h, (0, 2, 3, 1))
    h = self.block2(h, training=training)
    h = self.block3(h, training=training)
    h = self.block4(h, training=training)
    h = self.block5(h, training=training)
    h = self.block6(h, training=training)
    h = self.bn_7(h, training=training)
    h = self.relu(h)
    h = self.conv_7(h)
    h = keras.activations.tanh(h)

    return h


#################
# ## BIG GAN ## #
#################

# biggan architecture for large-scale ebm and generator

# TF2 Keras reimplementation of Pytorch BigGAN code from Hugging Face
# https://github.com/huggingface/pytorch-pretrained-BigGAN
# Original Code: Copyright (c) 2019 Thomas Wolf under MIT License

def snconv2d(**kwargs):
  return tfa.layers.SpectralNormalization(keras.layers.Conv2D(**kwargs))

def snlinear(**kwargs):
  return tfa.layers.SpectralNormalization(keras.layers.Dense(**kwargs))

class SelfAttn(keras.layers.Layer):
  """ Self attention Layer"""
  def __init__(self, in_channels):
    super().__init__()
    self.in_channels = in_channels
    self.snconv1x1_theta = snconv2d(filters=in_channels//8, kernel_size=1, use_bias=False)
    self.snconv1x1_phi = snconv2d(filters=in_channels//8, kernel_size=1, use_bias=False)
    self.snconv1x1_g = snconv2d(filters=in_channels//2, kernel_size=1, use_bias=False)
    self.snconv1x1_o_conv = snconv2d(filters=in_channels, kernel_size=1, use_bias=False)

    self.maxpool = keras.layers.MaxPooling2D()
    self.softmax  = keras.layers.Softmax()
    self.gamma = self.add_weight(name='gamma', shape=(1,), trainable=True)

  def call(self, x, training=False):
    _, h, w, ch = tuple(x.shape)

    # Theta path
    theta = self.snconv1x1_theta(x, training=training)
    theta = tf.reshape(theta, [-1, h*w, ch//8])
    # Phi path
    phi = self.snconv1x1_phi(x, training=training)
    phi = self.maxpool(phi)
    phi = tf.reshape(phi, [-1, h*w//4, ch//8])
    # Attn map
    attn = tf.matmul(theta, phi, transpose_b=True)
    attn = self.softmax(attn)
    # g path
    g = self.snconv1x1_g(x, training=training)
    g = self.maxpool(g)
    g = tf.reshape(g, (-1, h*w//4, ch // 2))
    # Attn_g - o_conv
    attn_g = tf.matmul(attn, g)
    attn_g = tf.reshape(attn_g, (-1, h, w, ch//2))
    attn_g = self.snconv1x1_o_conv(attn_g, training=training)
    # Out
    out = x + self.gamma * attn_g

    return out

class BigGANBatchNorm(keras.layers.Layer):
  def __init__(self, num_features, n_stats=51, epsilon=1e-4, conditional=True):
    super().__init__()
    self.num_features = num_features
    self.eps = epsilon
    self.conditional = conditional

    self.means = self.add_weight(name='mean', shape=(n_stats, num_features), trainable=False)
    self.vars = self.add_weight(name='var', shape=(n_stats, num_features), trainable=False)
    self.step_size = 1.0 / (n_stats - 1)

    if conditional:
      self.scale = snlinear(units=num_features, use_bias=False)
      self.offset = snlinear(units=num_features, use_bias=False)
    else:
      self.weight = self.add_weight(name='w', shape=(num_features,), trainable=False)
      self.bias = self.add_weight(name='b', shape=(num_features,), trainable=False)

  def call(self, x, training=False, truncation=None, cond_vector=None):
    # Retreive pre-computed statistics associated to this truncation
    coef, start_idx = math.modf(truncation / self.step_size)
    start_idx = int(start_idx)
    if coef != 0.0:  # Interpolate
      running_mean = self.means[start_idx] * coef + self.means[start_idx + 1] * (1 - coef)
      running_var = self.vars[start_idx] * coef + self.vars[start_idx + 1] * (1 - coef)
    else:
      running_mean = self.means[start_idx]
      running_var = self.vars[start_idx]

    #TODO: update running means/vars while in training mode

    if self.conditional:
      running_mean = tf.reshape(running_mean, [1, 1, 1, -1])
      running_var = tf.reshape(running_var, [1, 1, 1, -1])

      weight = 1 + tf.reshape(self.scale(cond_vector, training=training), [-1, 1, 1, self.num_features])
      bias = tf.reshape(self.offset(cond_vector, training=training), [-1, 1, 1, self.num_features])

      out = (x - running_mean) / tf.math.sqrt(running_var + self.eps) * weight + bias
    else:
      out = (x - running_mean) / tf.math.sqrt(running_var + self.eps) * self.weight + self.bias

    return out

# discriminator block for ebm layers
class EBMBlockBigGAN(keras.layers.Layer):
  def __init__(self, in_size, out_size, reduction_factor=4, down_sample=False):
    super().__init__()
    self.down_sample = down_sample
    middle_size = in_size // reduction_factor

    self.conv_0 = snconv2d(filters=middle_size, kernel_size=1)
    self.conv_1 = snconv2d(filters=middle_size, kernel_size=3, padding="SAME")
    self.conv_2 = snconv2d(filters=middle_size, kernel_size=3, padding="SAME")
    self.conv_3 = snconv2d(filters=out_size, kernel_size=1)

    self.relu = keras.activations.relu

    if self.down_sample:
      self.downsampling_layer = keras.layers.AveragePooling2D()

    self.learnable_sc = True if (in_size != out_size) else False
    if self.learnable_sc:
      self.conv_sc = snconv2d(filters=out_size-in_size, kernel_size=1)

  def shortcut(self, x, training=False):
    if self.learnable_sc:
      return tf.concat((x, self.conv_sc(x)), -1)
    else:
      return x

  def call(self, x, training=False):
    x0 = x

    x = self.relu(x)
    x = self.conv_0(x, training=training)
    x = self.relu(x)
    x = self.conv_1(x, training=training)
    x = self.relu(x)
    x = self.conv_2(x, training=training)
    x = self.relu(x)
    if self.down_sample:
      x = self.downsampling_layer(x)
    x = self.conv_3(x, training=training)

    if self.down_sample:
      x0 = self.downsampling_layer(x0)

    out = x + self.shortcut(x0)
    return out

# unconditional biggan discriminator for ebm
class EBMBigGAN(keras.Model):
  def __init__(self, im_sz=128, z_sz=128, channel_width=48):
    super().__init__()
    self.ch = channel_width

    if (im_sz == 256) or (im_sz == 224):
      layer_params = [
        (True, 1, 2),
        (False, 2, 2),
        (True, 2, 4),
        (False, 4, 4),
        (True, 4, 8),
        (False, 8, 8),
        (True, 8, 8),
        (False, 8, 8),
        (True, 8, 16),
        (False, 16, 16),
        (True, 16, 16),
        (False, 16, 16)
      ]
      attn_layer_pos = 4

    elif im_sz == 128:
      layer_params = [
        (True, 1, 2),
        (False, 2, 2),
        (True, 2, 4),
        (False, 4, 4),
        (True, 4, 8),
        (False, 8, 8),
        (True, 8, 16),
        (False, 16, 16),
        (True, 16, 16),
        (False, 16, 16)
      ]
      attn_layer_pos = 2

    elif im_sz == 64:
      layer_params = [
        (False, 1, 1),
        (True, 1, 2),
        (False, 2, 2),
        (True, 2, 4),
        (False, 4, 4),
        (True, 4, 8),
        (False, 8, 8),
        (True, 8, 16),
        (False, 16, 16)
      ]
      attn_layer_pos = 1

    else:
      raise ValueError('Invalid im_sz for BigGAN creation')

    self.rgb_conv = snconv2d(filters=self.ch, kernel_size=3, padding="SAME")
    layers = []
    for i, layer in enumerate(layer_params):
      if i == attn_layer_pos:
        layers.append(SelfAttn(self.ch*layer[1]))
      layers.append(EBMBlockBigGAN(self.ch*layer[1], self.ch*layer[2], down_sample=layer[0]))
    self.layers_out = layers

    self.pooling = keras.layers.GlobalAveragePooling2D()
    self.linear_head = snlinear(units=1, use_bias=False)

  def call(self, x, training=False):

    x = self.rgb_conv(x, training=training)

    for i, layer in enumerate(self.layers_out):
      if isinstance(layer, EBMBlockBigGAN):
        x = layer(x, training=training)
      else:
        x = layer(x, training=training)

    x = keras.activations.relu(x)
    x = self.pooling(x)
    
    x = self.linear_head(x)

    return x

# gen block for biggan
class GenBlockBigGAN(keras.layers.Layer):
  def __init__(self, in_size, out_size, make_bn, reduction_factor=4, up_sample=False, eps=1e-4, 
               n_stats=None, conditional=False):
    super().__init__()
    self.up_sample = up_sample
    self.drop_channels = (in_size != out_size)
    middle_size = in_size // reduction_factor

    bn_0_args, bn_args = {}, {}
    if n_stats is not None:
      bn_0_args = {'num_features': in_size, 'n_stats': n_stats, 'conditional': conditional}
      bn_args = {'num_features': middle_size, 'n_stats': n_stats, 'conditional': conditional}

    self.bn_0 = make_bn(epsilon=eps, **bn_0_args)
    self.conv_0 = snconv2d(filters=middle_size, kernel_size=1)

    self.bn_1 = make_bn(epsilon=eps, **bn_args)
    self.conv_1 = snconv2d(filters=middle_size, kernel_size=3, padding="SAME")

    self.bn_2 = make_bn(epsilon=eps, **bn_args)
    self.conv_2 = snconv2d(filters=middle_size, kernel_size=3, padding="SAME")

    self.bn_3 = make_bn(epsilon=eps, **bn_args)
    self.conv_3 = snconv2d(filters=out_size, kernel_size=1)

    self.relu = keras.activations.relu

    if self.up_sample:
      self.upsampling_layer = keras.layers.UpSampling2D()

  def call(self, x, training=False, truncation=None, cond_vector=None):
    x0 = x

    bn_args_call = {}
    if cond_vector is not None:
      bn_args_call = {'truncation': truncation, 'cond_vector': cond_vector}

    x = self.bn_0(x, training=training, **bn_args_call)
    x = self.relu(x)
    x = self.conv_0(x, training=training)

    x = self.bn_1(x, training=training, **bn_args_call)
    x = self.relu(x)
    if self.up_sample:
      x = self.upsampling_layer(x)
    x = self.conv_1(x, training=training)

    x = self.bn_2(x, training=training, **bn_args_call)
    x = self.relu(x)
    x = self.conv_2(x, training=training)

    x = self.bn_3(x, training=training, **bn_args_call)
    x = self.relu(x)
    x = self.conv_3(x, training=training)

    if self.drop_channels:
      new_channels = x0.shape[3] // 2
      x0 = x0[:, :, :, :new_channels]
    if self.up_sample:
      x0 = self.upsampling_layer(x0)

    out = x + x0
    return out

# unconditional biggan generator
class GenBigGAN(keras.Model):
  def __init__(self, im_sz=128, z_sz=128, channel_width=128, col_ch=3, eps=1e-4):
    super().__init__()
    self.ch = channel_width
    self.z_sz = z_sz
    self.col_ch = col_ch

    if im_sz == 128:
      # default parameters: 128x128 biggan deep
      layer_params = [
        (False, 16, 16),
        (True, 16, 16),
        (False, 16, 16),
        (True, 16, 8),
        (False, 8, 8),
        (True, 8, 4),
        (False, 4, 4),
        (True, 4, 2),
        (False, 2, 2),
        (True, 2, 1)
      ]
      attn_layer_pos = 8
      self.base_ch = 16 * self.ch

    elif im_sz == 64:
      # default parameters: 128x128 biggan deep
      layer_params = [
        (False, 16, 16),
        (True, 16, 16),
        (False, 16, 16),
        (True, 16, 8),
        (False, 8, 8),
        (True, 8, 4),
        (False, 4, 4),
        (True, 4, 2),
        (False, 2, 2)
      ]
      attn_layer_pos = 8
      self.base_ch = 16 * self.ch

    else:
      raise ValueError('Invalid im_sz for BigGAN creation')

    def make_bn(**kwargs):
      return keras.layers.experimental.SyncBatchNormalization(**kwargs)

    self.gen_z = snlinear(units=4 * 4 * self.base_ch)
    layers = []
    for i, layer in enumerate(layer_params):
      if i == attn_layer_pos:
        layers.append(SelfAttn(self.ch*layer[1]))
      layers.append(GenBlockBigGAN(self.ch*layer[1],
                                   self.ch*layer[2],
                                   make_bn,
                                   up_sample=layer[0],
                                   eps=eps))
    self.layers_out = layers

    self.bn = make_bn(epsilon=eps)
    self.conv_to_rgb = snconv2d(filters=self.ch, kernel_size=3, padding="SAME")

  def generate_latent_z(self, num_ims, truncation=1):
    return truncation * tf.random.truncated_normal([num_ims, self.z_sz])  # noise sample

  def generate_images(self, num_ims, truncation=1):
    # noise and label
    z = self.generate_latent_z(num_ims, truncation)
    return self.call(z)

  def call(self, z_init, training=True):

    z = self.gen_z(z_init, training=training)
    z = tf.reshape(z, (-1, 4, 4, self.base_ch))

    for i, layer in enumerate(self.layers_out):
      if isinstance(layer, GenBlockBigGAN):
        z = layer(z, training=training)
      else:
        z = layer(z, training=training)

    z = self.bn(z, training=training)
    z = keras.activations.relu(z)
    z = self.conv_to_rgb(z, training=training)
    z = z[:, :, :, 0:self.col_ch]
    z = keras.activations.tanh(z)

    return z

# conditional biggan generator
class GenBigGANCond(keras.Model):
  def __init__(self, im_sz=128, z_sz=128, channel_width=128, n_stats=51, eps=1e-4, num_classes=1000):
    super().__init__()
    self.ch = channel_width
    self.n_stats = n_stats
    self.z_sz = z_sz
    self.num_classes = num_classes

    # patch for generating resized 224x224 images from 256x256 images
    if im_sz == 224:
      im_sz = 256
      self.resize = 224
    elif im_sz == 64:
      im_sz = 128
      self.resize = 64
    else:
      self.resize = None

    if im_sz == 128: 
      layer_params = [(False, 16, 16),
                      (True, 16, 16),
                      (False, 16, 16),
                      (True, 16, 8),
                      (False, 8, 8),
                      (True, 8, 4),
                      (False, 4, 4),
                      (True, 4, 2),
                      (False, 2, 2),
                      (True, 2, 1)]
    elif im_sz == 256: 
      layer_params = [(False, 16, 16),
                      (True, 16, 16),
                      (False, 16, 16),
                      (True, 16, 8),
                      (False, 8, 8),
                      (True, 8, 8),
                      (False, 8, 8),
                      (True, 8, 4),
                      (False, 4, 4),
                      (True, 4, 2),
                      (False, 2, 2),
                      (True, 2, 1)]
    elif im_sz == 512: 
      layer_params = [(False, 16, 16),
                      (True, 16, 16),
                      (False, 16, 16),
                      (True, 16, 8),
                      (False, 8, 8),
                      (True, 8, 8),
                      (False, 8, 8),
                      (True, 8, 4),
                      (False, 4, 4),
                      (True, 4, 2),
                      (False, 2, 2),
                      (True, 2, 1),
                      (False, 1, 1),
                      (True, 1, 1)]
    else:                  
      raise ValueError('Invalid im_sz. Use 128, 256, or 512.')

    attn_layer_pos = 8
    self.base_ch = 16 * self.ch

    def make_bn(**kwargs):
      return BigGANBatchNorm(**kwargs)

    self.embeddings = keras.layers.Dense(units=z_sz, use_bias=False)
    self.gen_z = snlinear(units=4 * 4 * self.base_ch)

    layers = []
    for i, layer in enumerate(layer_params):
      if i == attn_layer_pos:
        layers.append(SelfAttn(self.ch*layer[1]))
      layers.append(GenBlockBigGAN(self.ch*layer[1],
                                   self.ch*layer[2],
                                   make_bn,
                                   up_sample=layer[0],
                                   n_stats=self.n_stats, 
                                   conditional=True,
                                   eps=eps))
    self.layers_out = layers

    self.bn = make_bn(num_features=self.ch, n_stats=self.n_stats, epsilon=eps, conditional=False)
    self.conv_to_rgb = snconv2d(filters=self.ch, kernel_size=3, padding="SAME")

  def generate_latent_z(self, num_ims, truncation=1):
    return truncation * tf.random.truncated_normal([num_ims, self.z_sz])  # noise sample

  def generate_class_label(self, num_ims):
    y_index = tf.random.uniform([num_ims], maxval=self.num_classes, dtype=tf.int32)
    return tf.one_hot(y_index, self.num_classes)  # one-hot ImageNet label

  def generate_images(self, num_ims, truncation=1):
    # noise and label
    z = self.generate_latent_z(num_ims, truncation)
    y = self.generate_class_label(num_ims)
    return self.call(z, y, truncation)

  def call(self, z_init, y=None, truncation=1, training=False):
    assert 0 < truncation <= 1

    if y is None:
      # draw random label
      y = self.generate_class_label(tf.shape(z_init)[0])

    embed = self.embeddings(y)
    cond_vector = tf.concat((z_init, embed), axis=1)

    z = self.gen_z(cond_vector, training=training)
    z = tf.reshape(z, (-1, 4, 4, 16 * self.ch))

    for i, layer in enumerate(self.layers_out):
      if isinstance(layer, GenBlockBigGAN):
        z = layer(z, training=training, truncation=truncation, cond_vector=cond_vector)
      else:
        z = layer(z, training=training)

    z = self.bn(z, training=training, truncation=truncation)
    z = keras.activations.relu(z)
    z = self.conv_to_rgb(z, training=training)
    z = z[:, :, :, 0:3]
    z = keras.activations.tanh(z)

    if self.resize is not None:
      z = tf.image.resize(z, [self.resize, self.resize])

    return z


########################
# ## WRN CLASSIFIER ## #
########################

# TF2 Keras reimplementation of Pytorch WideResNet code from:
# https://github.com/meliketoy/wide-resnet.pytorch
# Original Code: Copyright (c) 2018 Bumsoo Kim under MIT License

def conv_init(clf):
  for param in clf.trainable_variables:
    if len(param.shape) == 4:
      param *= 2**0.5

class ResBlock(keras.layers.Layer):
  def __init__(self, in_channels, out_channels, hidden_channels=None, stride=1, dropout=0.0):
    super(ResBlock, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.stride = stride
    self.hidden_channels = hidden_channels if hidden_channels is not None else out_channels
    self.learnable_sc = in_channels != out_channels or stride != 1

    self.conv_1 = keras.layers.Conv2D(filters=self.hidden_channels, kernel_size=3, padding="SAME")
    if stride == 1:
      self.conv_2 = keras.layers.Conv2D(filters=out_channels, kernel_size=3, strides=stride, padding="SAME")
    else:
      self.conv_2 = keras.layers.Conv2D(filters=out_channels, kernel_size=3, strides=stride, padding="VALID")

    self.bn_1 = keras.layers.experimental.SyncBatchNormalization(epsilon=1e-5, momentum=0.1)
    self.bn_2 = keras.layers.experimental.SyncBatchNormalization(epsilon=1e-5, momentum=0.1)

    self.dropout = keras.layers.Dropout(0.0)

    self.relu = keras.activations.relu

    if self.learnable_sc:
      self.shortcut = keras.layers.Conv2D(filters=out_channels, kernel_size=1, strides=stride, padding="VALID")

  def call(self, x, training):

    out = self.dropout(self.conv_1(keras.activations.relu(self.bn_1(x, training=training))))
    out = keras.activations.relu(self.bn_2(out, training=training))
    if self.stride == 2:
      out = tf.pad(out, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]))
    out = self.conv_2(out)
    if self.learnable_sc:
      out += self.shortcut(x)
    else:
      out += x

    return out

class WideResNetTF2(keras.Model):
  def __init__(self, depth=28, widen_factor=10, num_classes=10, dropout=0.0):
    super().__init__()

    self.in_planes = 16
    self.dropout = dropout

    assert ((depth-4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
    n = (depth-4)/6
    k = widen_factor
    nStages = [16, 16*k, 32*k, 64*k]

    self.conv1 = keras.layers.Conv2D(filters=nStages[0], kernel_size=3, padding="SAME")
    self.layer1 = self._wide_layer(ResBlock, nStages[1], n, stride=1)
    self.layer2 = self._wide_layer(ResBlock, nStages[2], n, stride=2)
    self.layer3 = self._wide_layer(ResBlock, nStages[3], n, stride=2)
    self.bn1 = keras.layers.experimental.SyncBatchNormalization(epsilon=1e-5, momentum=0.9)
    self.pool = keras.layers.GlobalAveragePooling2D()
    self.linear = keras.layers.Dense(num_classes)

  def _wide_layer(self, block, planes, num_blocks, stride):
    strides = [stride] + [1]*(int(num_blocks)-1)
    layers = []

    for stride in strides:
      layers.append(block(self.in_planes, planes, stride=stride, dropout=self.dropout))
      self.in_planes = planes

    return layers

  def call(self, x, training=True):
    out = self.conv1(x)
    for sublayer in self.layer1:
      out = sublayer(out, training=training)
    for sublayer in self.layer2:
      out = sublayer(out, training=training)
    for sublayer in self.layer3:
      out = sublayer(out, training=training)
    out = keras.activations.relu(self.bn1(out, training=training))
    out = self.pool(out)
    out = self.linear(out)

    return out


###################################
# ## FUNCTIONS TO GET NETWORKS ## #
###################################

# function to make ebm network
def create_ebm(config):
  inputs = tf.keras.Input(shape=config['image_dims'])
  if config['net_type'] == 'ebm_sngan':
    if config['image_dims'][0] == 32:
      model = EBMSNGAN32()
    elif config['image_dims'][0] == 64:
      model = EBMSNGAN64()
    elif config['image_dims'][0] == 128:
      model = EBMSNGAN128()
    else:
      raise ValueError('Invalid image_dims for ebm_sngan')
  elif config['net_type'] == 'ebm_biggan':
    model = EBMBigGAN(im_sz=config['image_dims'][0])
  else:
    raise ValueError('Invalid net_type')
  outputs = model(inputs, training=False)
  return tf.keras.models.Model(inputs, outputs)

# function to make gen network
def create_gen(config):
  if config['gen_type'] == 'gen_sngan':
    if config['image_dims'][0] == 32:
      return GenSNGAN32()
    elif config['image_dims'][0] == 64:
      return GenSNGAN64()
    elif config['image_dims'][0] == 128:
      return GenSNGAN128()
    else:
      raise ValueError('Invalid image_dims for sngan ebm net')
  elif config['gen_type'] == 'gen_biggan':
    return GenBigGAN(config['image_dims'][0], col_ch=config['image_dims'][2])
  elif config['gen_type'] == 'gen_biggan_cond':
    return GenBigGANCond(im_sz=config['image_dims'][0])
  else:
    raise ValueError('Invalid gen_type')
