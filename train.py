import os
from time import time
from datetime import datetime
import importlib

import tensorflow as tf

from data import get_dataset
from utils import setup_exp, save_model, plot_ims, plot_diagnostics
from init import init_strategy, initialize_nets_and_optim, initialize_persistent, initialize_data
from init import determinism_test, StepScheduleMCMC

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('config_name', default='config', help='config file. Default is config.py file.')
args = parser.parse_args()


###############
# ## SETUP ## #
###############

# get experiment config
config_module = importlib.import_module(args.config_name.replace('/', '.')[:-3])
config = config_module.config

# give exp_name unique timestamp identifier
time_str = datetime.now().strftime('%y-%m-%d-%H-%M-%S')
config['exp_name'] = config['exp_name'] + '_' + time_str

# setup folders, save code, set seed and get device
setup_exp(os.path.join(config['exp_dir'], config['exp_name']), 
          ['checkpoints', 'shortrun', 'plots'], 
          [os.path.join(config['root_path'], code_file) for code_file in
          ['train.py', 'nets.py', 'utils.py', 'data.py', 'init.py', args.config_name]],
          config['gs_path'])

# initialize distribution strategy
strategy = init_strategy(config)


##################################################
# ## INITIALIZE NETS, DATA, PERSISTENT STATES ## #
##################################################

# load nets and optim
ebm, ebm_optim, gen, gen_optim = initialize_nets_and_optim(config, strategy)
# test deterministic ouput
determinism_test(config, strategy, ebm, gen)


# Calculate per replica batch size, and distribute the datasets
print('Importing data...')
per_replica_batch_size = config['batch_size'] // strategy.num_replicas_in_sync
tpu_tensor_size = [per_replica_batch_size] + config['image_dims']
# initialize data
train_iterator, init_iterator, data_samples = initialize_data(config, strategy, gen)

# plot example of data images
plot_ims(os.path.join(config['exp_dir'], config['exp_name'], 'shortrun/data.pdf'), 
         data_samples[0:per_replica_batch_size])

# initialize persistent states
persistent_states, persistent_z, burnin_count = initialize_persistent(config, strategy, gen)


###########################
# ## TF GRAPH BUILDERS ## #
###########################

def make_langevin_update(num_mcmc_steps):
  @tf.function
  def langevin_update(images):
    images_samp = tf.identity(images)

    # container for grad diagnostic
    grad_norm = tf.constant(0, dtype=tf.float32)

    # langevin updates
    for i in tf.range(num_mcmc_steps):
      with tf.GradientTape() as tape:
        tape.watch(images_samp)
        energy = tf.math.reduce_sum(ebm(images_samp, training=False)) / config['mcmc_temp']
        if 'tau' in config.keys() and config['tau'] > 0:
          energy += tf.math.reduce_sum(images_samp**2) / (2 * config['tau']**2)
      grads = tape.gradient(energy, images_samp)
      # clip gradient norm (set to large value that won't interfere with standard dynamics)
      if config['clip_langevin_grad']:
        grads = tf.clip_by_norm(grads, config['max_langevin_norm'] / ((config['epsilon'] ** 2) / 2), axes=[1, 2, 3])

      # update images
      images_samp -= ((config['epsilon'] ** 2) / 2) * grads
      images_samp += config['epsilon'] * tf.random.normal(shape=tpu_tensor_size)

      # record gradient norm
      grad_norm = tf.math.reduce_mean(tf.norm(tf.reshape(grads, shape=[per_replica_batch_size, -1]), axis=1))
      grad_norm *= ((config['epsilon'] ** 2) / 2)

    return images_samp, grad_norm

  return langevin_update

# tf graph for single training update
def make_train_step(num_mcmc_steps, burnin_steps, persistent_resample_prob):
  # make update for training langevin sequence
  langevin_update_train = make_langevin_update(num_mcmc_steps)

  # make update to burn in rejuvenated samples
  langevin_update_burnin = make_langevin_update(burnin_steps)

  # training update function
  @tf.function
  def train_step(images_data_in, persistent_init, persistent_z_init, burnin_count, images_data_init):

    def step_fn(images_data_in, persistent_init, persistent_z_init, burnin_count, images_data_init):

      # initialize mcmc samples
      if config['mcmc_init'].endswith('persistent'):
        # select random persistent states
        shuffled_indices = tf.random.shuffle(tf.range(0, tf.shape(persistent_init)[0]))[0:per_replica_batch_size]
        images_samp = tf.gather(persistent_init, shuffled_indices)
        burnin_count_batch = tf.gather(burnin_count, shuffled_indices)
        if config['mcmc_init'] == 'coop_persistent':
          # get persistent latent state
          z_samp = tf.gather(persistent_z_init, shuffled_indices)
      elif config['mcmc_init'] == 'coop':
        # initialize from the output of a generator
        z_samp = gen.generate_latent_z(per_replica_batch_size)
        images_samp = tf.identity(gen(z_samp, training=config['gen_batch_norm']))
      elif config['mcmc_init'] == 'data':
        images_samp = images_data_init
      else:
        raise ValueError('Invalid choice of mcmc_init')

      # rejuvenate samples
      if config['mcmc_init'].endswith('persistent'):

        # randomly choose states to rejuvenate
        update_inds = tf.random.uniform(shape=[per_replica_batch_size]) < persistent_resample_prob
        update_inds = tf.cast(update_inds, dtype=tf.float32)
        # also rejuvenate persistent states that have been updated many times
        if config['max_mcmc_updates'] > 0:
          max_samp_inds = tf.cast(burnin_count_batch > config['max_mcmc_updates'], tf.float32)
          update_inds = 1 - (1 - update_inds) * (1 - max_samp_inds)

        if config['mcmc_init'] == 'persistent':
          # rejuvenate from uniform noise
          update_inds = tf.reshape(update_inds, [-1, 1, 1, 1])
          random_states = 2 * tf.random.uniform(shape=tpu_tensor_size) - 1
          images_samp = images_samp * (1 - update_inds) + random_states * update_inds
        elif config['mcmc_init'] == 'data_persistent':
          # rejuvenate from uniform noise
          update_inds = tf.reshape(update_inds, [-1, 1, 1, 1])
          images_samp = images_samp * (1 - update_inds) + images_data_init * update_inds
        elif config['mcmc_init'] == 'coop_persistent':
          # rejuvenate latent z
          update_inds = tf.reshape(update_inds, [-1, 1])
          random_states = gen.generate_latent_z(per_replica_batch_size)
          z_samp = z_samp * (1 - update_inds) + random_states * update_inds

          # rejuvenate persistent images for updated z
          update_inds = tf.reshape(update_inds, [-1, 1, 1, 1])
          images_gen_new = gen(z_samp, training=config['gen_batch_norm'])
          images_samp = images_samp * (1 - update_inds) + images_gen_new * update_inds


      # burnin samples that have just been rejuvenated
      if burnin_steps > 0 and tf.math.reduce_sum(tf.cast(burnin_count_batch == 0, dtype=tf.float32)) > 0:
        images_burnin = tf.identity(tf.gather(images_samp, tf.reshape(tf.where(burnin_count_batch == 0), [-1])))
        images_burnin, _ = langevin_update_burnin(images_burnin)
        images_samp = tf.tensor_scatter_nd_update(images_samp, tf.where(burnin_count_batch == 0), images_burnin)

      # langevin updates
      images_samp, grad_norm = langevin_update_train(images_samp)


      # get data states to update ebm
      if config['data_type'] == 'generator':
        images_data = gen.generate_images(per_replica_batch_size)
      else:
        images_data = images_data_in
      # perturb data images with gaussian noise
      images_data_perturbed = images_data + config['data_epsilon'] * tf.random.normal(shape=tpu_tensor_size)

      # get loss and update ebm
      with tf.GradientTape() as tape:
        # energy of data and model samples
        en_pos = ebm(images_data_perturbed, training=False)
        en_neg = ebm(images_samp, training=False)
        # maximum likelihood 'loss'
        loss = (tf.math.reduce_mean(en_pos) - tf.math.reduce_mean(en_neg)) / config['mcmc_temp']
        # rescale to adjust for summation over number of model replicas
        loss_scaled = loss / strategy.num_replicas_in_sync
      # get gradients
      ebm_grads = tape.gradient(loss_scaled, ebm.trainable_variables)
      # clip gradient norm
      if config['clip_ebm_grad']:
        ebm_grads = [tf.clip_by_norm(g, config['max_grad_norm']) for g in ebm_grads]
      # update ebm
      ebm_optim.apply_gradients(list(zip(ebm_grads, ebm.trainable_variables)))

      # update generator
      if config['mcmc_init'].startswith('coop') and config['update_generator']:
        with tf.GradientTape() as tape:
          # reconstruction loss for sampled states
          gen_loss = tf.math.reduce_sum((images_samp - gen(z_samp, training=config['gen_batch_norm']))**2) / per_replica_batch_size
          # rescale to adjust for summation over number of replicas
          gen_loss_scaled = gen_loss / strategy.num_replicas_in_sync
        # get gradients
        gen_grads = tape.gradient(gen_loss_scaled, gen.trainable_variables)
        # clip gradient norm
        if config['clip_gen_grad']:
          gen_grads = [tf.clip_by_norm(g, config['max_grad_norm']) for g in gen_grads]
        # update generator
        gen_optim.apply_gradients(list(zip(gen_grads, gen.trainable_variables)))


      if config['mcmc_init'].endswith('persistent'):
        # update persistent image bank
        persistent_new = tf.tensor_scatter_nd_update(persistent_init, 
          tf.reshape(shuffled_indices, shape=[-1, 1]), tf.identity(images_samp))
        # update number of mcmc updates for persistent states
        burnin_count_batch = (burnin_count_batch + 1) * (1 - tf.reshape(update_inds, [-1,]))
        burnin_count_new = tf.tensor_scatter_nd_update(burnin_count, 
          tf.reshape(shuffled_indices, [-1, 1]), tf.identity(burnin_count_batch))

        if config['mcmc_init'] == 'coop_persistent':
          # update latent z
          persistent_z_new = tf.tensor_scatter_nd_update(persistent_z_init, 
            tf.reshape(shuffled_indices, shape=[-1, 1]), tf.identity(z_samp))
        else:
          persistent_z_new = persistent_z_init

      else:
        persistent_new = persistent_init
        persistent_z_new = persistent_z_init

      return images_samp, persistent_new, persistent_z_new, burnin_count_new, \
        tf.reshape(loss, shape=[1]), tf.reshape(grad_norm, shape=[1])

    # perform ebm update
    images_samp, persistent_new, persistent_z_new, burnin_count_new, loss_out, grad_norm_out = \
      strategy.run(step_fn, args=(images_data_in, persistent_init, persistent_z_init, burnin_count, images_data_init))

    return images_samp, persistent_new, persistent_z_new, burnin_count_new, loss_out, grad_norm_out

  return train_step


##################
# ## LEARNING ## #
##################

# containers for gradient and energy difference records
loss_rec = tf.zeros(shape=[0], dtype=tf.float32)
grad_norm_rec = tf.zeros(shape=[0], dtype=tf.float32)

# number of mcmc_steps, burnin steps, rejuvenate prob.
mcmc_info = (None, None, None)

# scheduler for mcmc updates
mcmc_step_schedule = StepScheduleMCMC(config['mcmc_step_info'])

# start timer
time_check = time()

# training loop
print('Starting the training loop.')
for step in range(config['num_training_steps']):

  # update number of mcmc steps at certain thresholds
  mcmc_info_new = mcmc_step_schedule(step)
  if mcmc_info != mcmc_info_new:
    mcmc_info = mcmc_info_new
    train_update = make_train_step(int(mcmc_info[0]), int(mcmc_info[1]), float(mcmc_info[2]))

  # input data (from iterator or generator)
  if not config['data_type'] == 'generator':
    images_in = next(train_iterator)
  else:
    images_in = train_iterator
  # data to update persistent states ('data' or 'data_persistent' init only)
  if config['mcmc_init'].startswith('data'):
    images_data_init = next(init_iterator)
  else:
    images_data_init = init_iterator

  # training step on tf graph
  ims_samp, persistent_states, persistent_z, burnin_count, loss_out, grad_norm_out = \
    train_update(images_in, persistent_states, persistent_z, burnin_count, images_data_init)

  # update diagnostic records
  loss_gather = tf.math.reduce_mean(strategy.gather(loss_out, axis=0))
  grad_norm_gather = tf.math.reduce_mean(strategy.gather(grad_norm_out, axis=0))
  loss_rec = tf.concat([loss_rec, tf.reshape(loss_gather, shape=[1])], 0)
  grad_norm_rec = tf.concat([grad_norm_rec, tf.reshape(grad_norm_gather, shape=[1])], 0)

  # print and plot diagnostics
  if step == 0 or (step + 1) % config['info_freq'] == 0:
    print('Step: {}/{}'.format(step + 1, config['num_training_steps']))
    print('Energy Diff: {:.5f}   Grad Norm: {:.5f}'.format(loss_gather, grad_norm_gather))
    if step > 0:
      plot_diagnostics(config, step, loss_rec, grad_norm_rec)
      print('Time per Batch: {:.2f}'.format((time() - time_check) / config['info_freq']))
      time_check = time()

  # save images and checkpoints
  if step == 0 or (step + 1) % config['log_freq'] == 0:
    save_model(config, step, strategy, ebm, ebm_optim, ims_samp, persistent_states, 
               persistent_z, gen, gen_optim)
