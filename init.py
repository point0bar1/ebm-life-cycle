import tensorflow as tf
import tensorflow_addons as tfa

from nets import create_ebm, create_gen
from data import get_dataset
from utils import download_blob

import os
import pickle


def init_strategy(config):
  if config['device_type'] == 'tpu':
    # Set up TPU Distribution (set to run from Cloud TPU VM)
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
    #tf.config.experimental_connect_to_cluster(resolver)
    # This is the TPU initialization code that has to be at the beginning.
    tf.tpu.experimental.initialize_tpu_system(resolver)
    print("All devices: ", tf.config.list_logical_devices('TPU'))
    strategy = tf.distribute.TPUStrategy(resolver)
  elif config['device_type'] == 'gpu':
    # Set up GPU Distribution
    num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
    strategy = tf.distribute.MirroredStrategy(['GPU:'+str(i) for i in range(num_gpus)], 
                                              cross_device_ops=tf.distribute.NcclAllReduce())
  else:
    # default cpu strategy
    strategy = tf.distribute.OneDeviceStrategy('CPU:0')
  return strategy


def initialize_nets_and_optim(config, strategy):

  # Create the model, optimizer and metrics inside strategy scope, so that the
  # variables can be mirrored on each device.
  with strategy.scope():
    # set up ebm and optinally load weights
    ebm = create_ebm(config)
    if 'ebm_weights' in config.keys() and config['ebm_weights'] is not None:
      ebm.load_weights(config['ebm_weights'])

    # set up ebm optimizer
    if 'ebm_optim_type' in config.keys():
      # set up ebm optimizer schedule
      ebm_lr = StepScheduleLR(config['ebm_lr_info'])

      # set up ebm optim
      if config['ebm_optim_type'] == 'adam':
        ebm_optim = tf.keras.optimizers.Adam(learning_rate=ebm_lr, beta_1=0.9, beta_2=0.999)
      elif config['ebm_optim_type'] == 'sgd':
        ebm_optim = tf.keras.optimizers.SGD(learning_rate=ebm_lr)
      else:
        raise RuntimeError('Invalid ebm_optim_type')

      # moving average for weights
      ebm_optim = tfa.optimizers.MovingAverage(ebm_optim, average_decay=config['ebm_opt_decay'])
    else:
      ebm_optim = None

    # generator network for mcmc initialization
    if config['mcmc_init'].startswith('coop'):
      # set up generator model and optionally load weights
      gen = create_gen(config)
      if 'gen_weights' in config.keys() and config['gen_weights'] is not None:
        gen.load_weights(config['gen_weights'])

      if 'gen_optim_type' in config.keys() and config['update_generator']:
        # set up generator optimizer schedule
        gen_lr = StepScheduleLR(config['gen_lr_info'])

        # set up gen optimizer
        if config['gen_optim_type'] == 'adam':
          gen_optim = tf.keras.optimizers.Adam(learning_rate=gen_lr, beta_1=0.9, beta_2=0.999)
        elif config['gen_optim_type'] == 'sgd':
          gen_optim = tf.keras.optimizers.SGD(learning_rate=gen_lr)
        else:
          raise RuntimeError('Invalid gen_optim_type')

        # moving average for weights
        gen_optim = tfa.optimizers.MovingAverage(gen_optim, average_decay=config['gen_opt_decay'])
      else:
        gen.trainable = False
        gen_optim = None

    else:
      gen, gen_optim = None, None

  # function to download optim weights saved on cloud
  def download_optim(file_name):
    exp_folder = os.path.join(config['exp_dir'], config['exp_name'])
    # download persistent ims from cloud
    download_blob(config['gs_path'],
                  file_name,
                  os.path.join(exp_folder, 'checkpoints/optim_download.ckpt'))
    optim_weights = pickle.load(open(os.path.join(exp_folder, 'checkpoints/optim_download.ckpt'), 'rb'))
    # remove to save space
    os.remove(os.path.join(exp_folder, 'checkpoints/optim_download.ckpt'))

    return optim_weights

  # optim weight initialization (optional)
  if 'ebm_optim_weights' in config.keys() and config['ebm_optim_weights'] is not None: 
    # dummy function to initialize ebm optim
    @tf.function
    def initialize_ebm_optim():
      ebm_out = ebm(tf.random.normal(shape=[3]+config['image_dims']))
      null_grads_ebm = [tf.zeros_like(var) for var in ebm.trainable_variables]
      ebm_optim.apply_gradients(list(zip(null_grads_ebm, ebm.trainable_variables)))
    strategy.run(initialize_ebm_optim)

    # load ebm optim weights
    ebm_optim_weights = download_optim(config['ebm_optim_weights'])
    ebm_optim.set_weights(ebm_optim_weights)

  if gen_optim is not None:
    if 'gen_optim_weights' in config.keys() and config['gen_optim_weights'] is not None:
      # dummy function to initialize gen optim
      @tf.function
      def initialize_gen_optim():
        gen_out = gen(tf.random.normal(shape=[3, config['z_sz']]))
        null_grads_gen = [tf.zeros_like(var) for var in gen.trainable_variables]
        gen_optim.apply_gradients(list(zip(null_grads_gen, gen.trainable_variables)))
      strategy.run(initialize_gen_optim)

      # load gen optim weights
      gen_optim_weights = download_optim(config['gen_optim_weights'])
      gen_optim.set_weights(gen_optim_weights)

  return ebm, ebm_optim, gen, gen_optim


def initialize_data(config, strategy, gen=None, shuffle=True, repeat=True, get_label=False):

  per_replica_batch_size = config['batch_size'] // strategy.num_replicas_in_sync

  # distribute dataset across tpu's
  if not config['data_type'] == 'generator':
    train_dataset = strategy.distribute_datasets_from_function(
      lambda _: get_dataset(config, per_replica_batch_size, shuffle=shuffle, repeat=repeat, get_label=get_label))
    train_iterator = iter(train_dataset)
    # check data loader
    data_samples = strategy.gather(next(train_iterator), axis=0)
  else:
    def initialize_null_data(ctx):
      return tf.zeros([0], dtype=tf.float32)
    train_iterator = strategy.experimental_distribute_values_from_function(initialize_null_data)
    # get images from generator and plot
    data_samples = strategy.gather(gen.generate_images(num_ims=per_replica_batch_size), axis=0)

  # initialize iterator for mcmc initialization from data
  if 'mcmc_init' in config.keys() and config['mcmc_init'].startswith('data'):
    init_dataset = strategy.distribute_datasets_from_function(
      lambda _: get_dataset(config, per_replica_batch_size, shuffle=shuffle, repeat=repeat, get_label=get_label))
    init_iterator = iter(init_dataset)
  else:
    # dummy tensor for training where mcmc is not initialized from data samples (a.k.a. not CD)
    def initialize_null_data(ctx):
      return tf.zeros([0], dtype=tf.float32)
    init_iterator = strategy.experimental_distribute_values_from_function(initialize_null_data)

  return train_iterator, init_iterator, data_samples


def initialize_persistent(config, strategy, gen):

  exp_folder = os.path.join(config['exp_dir'], config['exp_name'])
  per_replica_batch_size = config['batch_size'] // strategy.num_replicas_in_sync

  # Calculate per replica persistent size, and distribute the persistent states
  print('Initializing persistent_states...')
  if config['mcmc_init'].endswith('persistent'):
    per_replica_persistent_size = config['persistent_size'] // strategy.num_replicas_in_sync
    per_replica_persistent_size = per_replica_batch_size * (per_replica_persistent_size // per_replica_batch_size)
    persistent_tensor_size = [per_replica_persistent_size] + config['image_dims']
    if config['mcmc_init'] == 'coop_persistent':
      persistent_latent_size = [per_replica_persistent_size, config['z_sz']]
    else:
      persistent_latent_size = [0]
  else:
    per_replica_persistent_size = 0
    persistent_tensor_size = [0]
    persistent_latent_size = [0]

  def initialize_persistent_states(ctx):
    if 'im_persistent_path' in config.keys() and config['im_persistent_path'] is not None:
      return persistent_im_tensor[(per_replica_persistent_size * ctx.replica_id_in_sync_group):
                                  (per_replica_persistent_size * (ctx.replica_id_in_sync_group + 1))]
    else:
      persistent_init_scale = config['persistent_init_scale'] if 'persistent_init_scale' in config else 1.0
      return persistent_init_scale * (2 * tf.random.uniform(shape=persistent_tensor_size) - 1)

  def initialize_persistent_z(ctx):
    if gen is not None:
      if 'z_persistent_path' in config.keys() and config['z_persistent_path'] is not None:
        return persistent_z_tensor[(per_replica_persistent_size * ctx.replica_id_in_sync_group):
                                  (per_replica_persistent_size * (ctx.replica_id_in_sync_group + 1))]
      else:
        return gen.generate_latent_z(per_replica_persistent_size)
    else:
      return tf.random.truncated_normal(shape=persistent_latent_size)

  def initialize_burnin_inds(ctx):
    return tf.zeros([per_replica_persistent_size], dtype=tf.float32)

  def download_persistent_ims():
    # download persistent ims from cloud
    download_blob(config['gs_path'],
                  config['im_persistent_path'],
                  os.path.join(exp_folder, 'checkpoints/persistent.ckpt'))
    persistent_im_tensor = pickle.load(open(os.path.join(exp_folder, 'checkpoints/persistent.ckpt'), 'rb'))
    # remove to save space
    os.remove(os.path.join(exp_folder, 'checkpoints/persistent.ckpt'))

    return persistent_im_tensor

  def download_persistent_z():
    # download persistent ims from cloud
    download_blob(config['gs_path'],
                config['z_persistent_path'],
                os.path.join(exp_folder, 'checkpoints/persistent_z.ckpt'))
    persistent_z_tensor = pickle.load(open(os.path.join(exp_folder, 'checkpoints/persistent_z.ckpt'), 'rb'))
    # remove to save space
    os.remove(os.path.join(exp_folder, 'checkpoints/persistent_z.ckpt'))

    return persistent_z_tensor

  if config['mcmc_init'] == 'coop_persistent':
    if 'z_persistent_path' in config.keys() and config['z_persistent_path'] is not None:
      # load persistent z from previous run
      persistent_z_tensor = download_persistent_z()
    # initialized persistent z
    persistent_z = strategy.experimental_distribute_values_from_function(initialize_persistent_z)

    # function to initialize persistent states from the output of the generator network
    @tf.function
    def initialize_persistent_gen_states(persistent_z_in, persistent_states_init):
      persistent_states_out = tf.identity(persistent_states_init)
      inds_init = tf.cast(tf.experimental.numpy.arange(0, per_replica_batch_size), dtype=tf.int32)
      for i in range(per_replica_persistent_size // per_replica_batch_size):
        inds_init = tf.experimental.numpy.arange(i*per_replica_batch_size, (i+1)*per_replica_batch_size)
        inds_init = tf.cast(inds_init, dtype=tf.int32)
        z_batch = tf.gather_nd(persistent_z_in, tf.reshape(inds_init, [-1, 1]))
        persistent_batch = gen.call(z_batch)
        persistent_states_out = tf.tensor_scatter_nd_update(persistent_states_out, 
            tf.reshape(inds_init, shape=[-1, 1]), tf.identity(persistent_batch))
      
      return persistent_states_out

    # initialize persistent states from output of generator
    if 'im_persistent_path' in config.keys() and config['im_persistent_path'] is not None:
      # load persistent images
      persistent_im_tensor = download_persistent_ims()
      persistent_states = strategy.experimental_distribute_values_from_function(initialize_persistent_states)
    else:
      # load persistent states to match persistent z
      persistent_states = strategy.experimental_distribute_values_from_function(initialize_persistent_states)
      persistent_states = strategy.run(initialize_persistent_gen_states, args=(persistent_z, persistent_states))

  elif config['mcmc_init'] == 'data_persistent':
    if 'im_persistent_path' in config.keys() and config['im_persistent_path'] is not None:
      # load saved checkpoint
      persistent_im_tensor = download_persistent_ims()
      persistent_states = strategy.experimental_distribute_values_from_function(initialize_persistent_states)
    else:
      # initialize persistent states from data samples
      train_dataset_mcmc_init = strategy.distribute_datasets_from_function(
        lambda _: get_dataset(config, per_replica_persistent_size))
      persistent_states = next(iter(train_dataset_mcmc_init))
    # null persistent z (unused)
    persistent_z = strategy.experimental_distribute_values_from_function(initialize_persistent_z)

  else:
    if 'im_persistent_path' in config.keys() and config['im_persistent_path'] is not None:
      # load saved checkpoint
      persistent_im_tensor = download_persistent_ims()
    persistent_states = strategy.experimental_distribute_values_from_function(initialize_persistent_states)
    persistent_z = strategy.experimental_distribute_values_from_function(initialize_persistent_z)

  # counts for number of mcmc updates for persistent states (used for rejuvenation)
  persistent_burnin = strategy.experimental_distribute_values_from_function(initialize_burnin_inds)

  return persistent_states, persistent_z, persistent_burnin


def determinism_test(config, strategy, ebm, gen):
  # test deterministic output of ebm
  with strategy.scope():
    z_test = tf.random.normal(shape=[3]+config['image_dims'])
    z_out_1 = ebm(z_test)
    z_out_2 = ebm(z_test[0:2])
  z_out_1 = strategy.gather(z_out_1, axis=0)
  z_out_2 = strategy.gather(z_out_2, axis=0)
  print('EBM Determinism Test (if deterministic, should be very close to 0): ', 
        tf.math.reduce_max(tf.math.abs(z_out_1[0] - z_out_2[0])))

  # test deterministic output of gen
  if gen is not None:
    with strategy.scope():
      gen_z = gen.generate_latent_z(3)
      gen_out_1 = gen(gen_z)
      gen_out_2 = gen(gen_z[0:2])
    gen_out_1 = strategy.gather(gen_out_1, axis=0)
    gen_out_2 = strategy.gather(gen_out_2, axis=0)
    print('Gen Determinism Test (if deterministic, should be very close to 0):', 
          tf.math.reduce_max(tf.math.abs(gen_out_1[0] - gen_out_2[0])))


#################################
# ## LEARNING RATE FUNCTIONS ## #
#################################

class StepScheduleLR(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, lr_info):
    super().__init__()

    self.lr_steps = tf.constant([lr_pair[0] for lr_pair in lr_info])
    self.lr_thresholds = tf.constant([lr_pair[1] for lr_pair in lr_info])
    self.lr_thresholds = tf.cast(self.lr_thresholds, dtype=tf.float32)

  def __call__(self, step):
    lr_ind = tf.math.reduce_max(tf.where(step >= self.lr_thresholds))
    return self.lr_steps[lr_ind]

class StepScheduleMCMC:
  def __init__(self, mcmc_step_info):

    self.mcmc_steps = tf.constant([mcmc_pair[0] for mcmc_pair in mcmc_step_info], dtype=tf.int32)
    self.burnin_steps = tf.constant([mcmc_pair[1] for mcmc_pair in mcmc_step_info], dtype=tf.int32)
    self.rejuvenate_info = tf.constant([mcmc_pair[2] for mcmc_pair in mcmc_step_info], dtype=tf.float32)

    self.mcmc_thresholds = tf.constant([mcmc_pair[3] for mcmc_pair in mcmc_step_info])
    self.mcmc_thresholds = tf.cast(self.mcmc_thresholds, dtype=tf.float32)

  def __call__(self, step):
    mcmc_ind = tf.math.reduce_max(tf.where(step >= self.mcmc_thresholds))
    return self.mcmc_steps[mcmc_ind], self.burnin_steps[mcmc_ind], self.rejuvenate_info[mcmc_ind]
