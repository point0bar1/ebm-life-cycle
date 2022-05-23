import tensorflow as tf
import tensorflow_datasets as tfds


# function to set up dataset
def get_dataset(config, batch_size, shuffle=True, repeat=True, get_label=False):
  # default dataset split is 'train' unless specified in config
  if 'split' in config.keys():
    split = config['split']
  else:
    split = 'train'

  # default to not use random crop unless specified in config
  if 'random_crop' in config.keys() and split == 'train':
    random_crop = config['random_crop']
  else:
    random_crop = False

  # path for loading data
  if config['gs_path'] is not None:
    data_dir = 'gs://' + config['gs_path']
    download = False
  else:
    data_dir = config['data_dir']
    download = config['data_download']

  # load tfrecords
  dataset = tfds.load(name=config['data_type'], split=split, data_dir=data_dir, 
                      download=download, try_gcs=False, with_info=False, as_supervised=False)

  def transform(features, scale_range=[1., 1.], aspect_range=[1., 1.], data_flip=False):
    # get image, cast to float, scale to [-1, 1] pixel range
    image = features['image']
    image = tf.cast(image, tf.float32)
    image = 2 * (image / 255.0) - 1

    if random_crop and config['data_type'] == 'cifar10':
      # cifar10 augmentation for WideResNet classifier training

      image = tf.pad(image, tf.constant([[4, 4,], [4, 4], [0, 0]]))
      image = tf.image.random_crop(image, config['image_dims'])
      data_flip = True

    elif random_crop:
      # function to replicate torchvision.transforms.RandomResizedCrop

      # get rescaled dimensions
      width, height = tf.shape(image)[0], tf.shape(image)[1]

      # get aspect of resized full image
      aspect_log = tf.math.log(tf.constant(aspect_range))
      target_aspect = tf.math.exp(tf.random.uniform(shape=[1], minval=aspect_log[0], maxval=aspect_log[1]))

      width_new = tf.cast(width, tf.float32) * tf.sqrt(target_aspect)
      height_new = tf.cast(height, tf.float32) / tf.sqrt(target_aspect)
      width_resize = (config['image_dims'][0] / tf.math.minimum(width_new, height_new)) * width_new
      height_resize = (config['image_dims'][1] / tf.math.minimum(width_new, height_new)) * height_new
      width_resize = tf.math.maximum(tf.cast(width_resize, dtype=tf.int32), tf.constant(config['image_dims'][0]))
      height_resize = tf.math.maximum(tf.cast(height_resize, dtype=tf.int32), tf.constant(config['image_dims'][1]))
      
      # rescale image according to target aspect to be a valid size for cropping
      image = tf.image.resize(image, tf.concat([width_resize, height_resize], 0), antialias=True)
      # select the scale of the patch to get from resized image
      scale = tf.math.sqrt(tf.random.uniform(shape=[1], minval=scale_range[0], maxval=scale_range[1]))
      scaled_dims = scale * tf.constant([config['image_dims'][0], config['image_dims'][1]], dtype=tf.float32)
      # get the patch according to the scaled dimensions
      crop_dims = tf.concat((tf.cast(scaled_dims, tf.int32), tf.constant([config['image_dims'][2]])), 0)
      image = tf.image.random_crop(image, crop_dims)

      # resize to the desired size
      image = tf.image.resize(image, (config['image_dims'][0], config['image_dims'][1]), antialias=True)

    elif image.shape[0:2] != config['image_dims'][0:2]:
      # center crop and resize

      # get rescaled dimensions
      width, height = tf.shape(image)[0], tf.shape(image)[1]

      # center crop
      if height > width:
        image = tf.image.crop_to_bounding_box(image, 0, (height - width) // 2, width, width)
      else:
        image = tf.image.crop_to_bounding_box(image, (width - height) // 2, 0, height, height)

      # resize to the desired size
      image = tf.image.resize(image, (config['image_dims'][0], config['image_dims'][1]), antialias=True)

    # left-right random flip
    if data_flip:
      image = tf.image.random_flip_left_right(image)

    # return image with or without label
    if not get_label:
      return image
    else:
      label = features['label']
      return image, label

  dataset = dataset.map(transform)

  if shuffle:
    # shuffle with buffer size
    dataset = dataset.shuffle(2500)
  if repeat:
    # infinite data loop
    dataset = dataset.repeat()
  dataset = dataset.batch(batch_size)

  return dataset
