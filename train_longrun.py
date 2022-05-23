import os
from time import time
from datetime import datetime
import importlib

import tensorflow as tf

from data import get_dataset
from utils import setup_exp, save_model, plot_ims, plot_diagnostics
from init import init_strategy, initialize_nets_and_optim, initialize_persistent, initialize_data
from init import determinism_test, StepScheduleMCMC
from nets import create_ebm

import matplotlib.pyplot as plt

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
          ['train_longrun.py', 'nets.py', 'utils.py', 'data.py', 'init.py', args.config_name]],
          config['gs_path'])

# initialize distribution strategy
strategy = init_strategy(config)


##################################################
# ## INITIALIZE NETS, DATA, PERSISTENT STATES ## #
##################################################

# load nets and optim
ebm, ebm_optim, gen, gen_optim = initialize_nets_and_optim(config, strategy)
if 'prior_weights' in config.keys() and config['prior_weights'] is not None:
  prior_ebm = create_ebm(config)
  prior_ebm.load_weights(config['prior_weights'])
else:
  prior_ebm = None
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

# initialize persistent states and burnin states
per_replica_persistent_size = config['persistent_size'] // strategy.num_replicas_in_sync
per_replica_burnin_size = config['burnin_size'] // strategy.num_replicas_in_sync
per_replica_persistent_size = per_replica_batch_size * (per_replica_persistent_size // per_replica_batch_size)
persistent_states, _, _ = initialize_persistent(config, strategy, gen)
# temporarily set new number of persistent states, get burnin states, reset config
persistent_size_old = config['persistent_size']
config['persistent_size'] = config['burnin_size']
burnin_states, _, _ = initialize_persistent(config, strategy, gen)
config['persistent_size'] = persistent_size_old


###########################
# ## TF GRAPH BUILDERS ## #
###########################

def langevin_update(images, mcmc_steps):
  images_samp = tf.identity(images)

  # container for grad diagnostic
  grad_norm = tf.constant(0, dtype=tf.float32)

  # langevin updates
  for i in tf.range(mcmc_steps):
    with tf.GradientTape() as tape:
      tape.watch(images_samp)
      energy = tf.math.reduce_sum(ebm(images_samp, training=False)) / config['mcmc_temp']
      if prior_ebm is not None:
        energy += tf.math.reduce_sum(prior_ebm(images_samp, training=False)) / config['prior_temp']
      if 'tau' in config.keys() and config['tau'] > 0:
        energy += tf.math.reduce_sum(images_samp**2) / (2 * config['tau']**2)
    grads = tape.gradient(energy, images_samp)
    # clip gradient norm (set to large value that won't interfere with standard dynamics)
    if config['clip_langevin_grad']:
      grads = tf.clip_by_norm(grads, config['max_langevin_norm'] / ((config['epsilon'] ** 2) / 2), axes=[1, 2, 3])

    # update images with gradient
    images_samp -= ((config['epsilon'] ** 2) / 2) * grads
    # update images with noise term (for early burnin samples, might only use the gradient and no noise)
    images_samp += config['epsilon'] * tf.random.normal(shape=tpu_tensor_size)

    # record gradient norm
    grad_norm = tf.math.reduce_mean(tf.norm(tf.reshape(grads, shape=[per_replica_batch_size, -1]), axis=1))
    grad_norm *= ((config['epsilon'] ** 2) / 2)

  return images_samp, grad_norm

def make_tf_graph(steps, burnin_steps, min_updates):
  @tf.function
  def step_fn_split(images_data_in, persistent_init, burnin_init, burnin_count, images_data_init):
    # indices for burnin states
    burnin_inds = tf.random.shuffle(tf.range(0, tf.shape(burnin_init)[0]))[0:per_replica_batch_size]
    # indices for update states
    update_inds = tf.random.shuffle(tf.range(0, tf.shape(persistent_init)[0]))[0:per_replica_batch_size]

    # burnin samples
    images_burnin = tf.gather(burnin_init, burnin_inds)
    counts_burnin = tf.gather(burnin_count, burnin_inds)
    # long run samples to update the ebm
    images_samp = tf.gather(persistent_init, update_inds)


    # langevin update to move states towards steady state
    images_burnin, _ = langevin_update(images_burnin, burnin_steps)

    # langevin updates
    images_samp, grad_norm = langevin_update(images_samp, steps)


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
      # rescale to adjust for summation over number of replicas
      loss_scaled = loss / strategy.num_replicas_in_sync
    # get gradients
    ebm_grads = tape.gradient(loss_scaled, ebm.trainable_variables)
    # clip gradient norm
    if config['clip_ebm_grad']:
      ebm_grads = [tf.clip_by_norm(g, config['max_grad_norm']) for g in ebm_grads]
    # update ebm
    ebm_optim.apply_gradients(list(zip(ebm_grads, ebm.trainable_variables)))


    # rejuvenate chains in long-run bank
    if config['mcmc_init'] == 'persistent':
      # rejuvenate from uniform noise
      persistent_init_scale = config['persistent_init_scale'] if 'persistent_init_scale' in config else 1.0
      random_states = persistent_init_scale * (2 * tf.random.uniform(shape=tpu_tensor_size) - 1)
    elif config['mcmc_init'] == 'data_persistent':
      # rejuvenate from data
      random_states = images_data_init
    elif config['mcmc_init'] == 'coop_persistent':
      # rejuvenate from generator
      random_states = gen(gen.generate_latent_z(per_replica_batch_size), training=config['gen_batch_norm'])
    else:
      raise ValueError('Invalid mcmc_init for longrun training (use persistent, coop_persistent, or data_persistent)')

    # choose states that have crossed update threshold and add to the update bank
    rejuv_inds = counts_burnin >= min_updates
    rejuv_inds = tf.reshape(tf.cast(rejuv_inds, dtype=tf.float32), [-1, 1, 1, 1])
    # add burnin samples that have been updated many times to the update bank, overwrite random old states
    images_samp_rejuv = images_samp * (1 - rejuv_inds) + images_burnin * rejuv_inds
    # update burnin images that have exceed count with newly rejuvenated samples
    images_burnin_rejuv = images_burnin * (1 - rejuv_inds) + random_states * rejuv_inds

    # update persistent image bank and burnin image bank
    persistent_init = tf.tensor_scatter_nd_update(persistent_init,
        tf.reshape(update_inds, shape=[-1, 1]), tf.identity(images_samp_rejuv))
    burnin_init = tf.tensor_scatter_nd_update(burnin_init, 
        tf.reshape(burnin_inds, shape=[-1, 1]), tf.identity(images_burnin_rejuv))
    # update number of mcmc updates for persistent states
    counts_new = (counts_burnin + 1) * (1 - tf.reshape(rejuv_inds, [-1,]))
    burnin_count = tf.tensor_scatter_nd_update(burnin_count, 
        tf.reshape(burnin_inds, [-1, 1]), tf.identity(counts_new))


    return images_samp, persistent_init, burnin_init, burnin_count, tf.reshape(loss, shape=[1]), tf.reshape(grad_norm, shape=[1])

  return step_fn_split

# tf graph for single training update
def make_train_step(steps, burnin_steps, min_updates):

  # make tf graph
  step_update_fn = make_tf_graph(steps, burnin_steps, min_updates)

  # training update function
  def train_step(persistent_init, burnin_init, burnin_count):

    # input data (from iterator or generator)
    if not config['data_type'] == 'generator':
      images_in = next(train_iterator)
    else:
      images_in = train_iterator

    # data to update persistent states ('data' or 'data_persistent' init only)
    if config['mcmc_init'].startswith('data'):
      images_update = next(init_iterator)
    else:
      images_update = init_iterator

    images_samp, persistent_new, burnin_new, burnin_count_new, loss_out, grad_norm_out = \
      strategy.run(step_update_fn, args=(images_in, persistent_init, burnin_init, burnin_count, images_update))

    return images_samp, persistent_new, burnin_new, burnin_count_new, loss_out, grad_norm_out

  return train_step


##################
# ## LEARNING ## #
##################

# containers for gradient and energy difference records
loss_rec = tf.zeros(shape=[0], dtype=tf.float32)
grad_norm_rec = tf.zeros(shape=[0], dtype=tf.float32)

# start timer
time_check = time()

# scheduler for mcmc updates
mcmc_step_schedule = StepScheduleMCMC(config['mcmc_step_info'])
# number of mcmc_steps, burnin steps, steps to include in update bank
mcmc_info = (None, None, None)

# training loop
print('Starting the training loop.')
for step in range(config['num_training_steps']):

  # update number of mcmc steps at certain thresholds
  mcmc_info_new = mcmc_step_schedule(step)
  if mcmc_info != mcmc_info_new:
    mcmc_info = mcmc_info_new
    train_update = make_train_step(int(mcmc_info[0]), int(mcmc_info[1]), int(mcmc_info[2]))

    # randomize burnin count to ensure steady stream of states into bank
    def randomize_burnin_count(ctx):
      random_count = tf.random.uniform(shape=[per_replica_burnin_size], minval=0, maxval=mcmc_info[2], dtype=tf.int32)
      return tf.cast(random_count, tf.float32)
    burnin_update_count = strategy.experimental_distribute_values_from_function(randomize_burnin_count)

  # training step on tf graph
  ims_samp, persistent_states, burnin_states, burnin_update_count, loss_out, grad_norm_out = \
    train_update(persistent_states, burnin_states, burnin_update_count)

  # update diagnostic records
  loss_gather = tf.math.reduce_mean(strategy.gather(loss_out, axis=0))
  grad_norm_gather = tf.math.reduce_mean(strategy.gather(grad_norm_out, axis=0))
  loss_rec = tf.concat([loss_rec, tf.reshape(loss_gather, shape=[1])], 0)
  grad_norm_rec = tf.concat([grad_norm_rec, tf.reshape(grad_norm_gather, shape=[1])], 0)

  # print and plot diagnostics
  if step == 0 or (step + 1) % config['info_freq'] == 0:
    print('Training Step: {}/{}'.format(step + 1, config['num_training_steps']))
    print('Energy Diff: {:.5f}   Grad Norm: {:.5f}'.format(loss_gather, grad_norm_gather))

    # visualize burnin update count histogram
    burnin_update_gather = strategy.gather(burnin_update_count, axis=0)
    plt.hist(burnin_update_gather.numpy())
    plt.savefig(os.path.join(config['exp_dir'], config['exp_name'], 'plots/burnin_hist.png'))
    plt.close()

    if step > 0:
      plot_diagnostics(config, step, loss_rec, grad_norm_rec)
      print('Time per Batch: {:.2f}'.format((time() - time_check) / config['info_freq']))
      time_check = time()

  # save images and checkpoints
  if step == 0 or (step + 1) % config['log_freq'] == 0:
    save_model(config, step, strategy, ebm, ebm_optim, ims_samp, persistent_states, None, gen, gen_optim)
