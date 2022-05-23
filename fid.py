# calculate unofficial TF2 FID score

import os
from datetime import datetime
import pickle
from tqdm import tqdm
import importlib

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input

from init import init_strategy, initialize_nets_and_optim, initialize_data, determinism_test
from data import get_dataset
from utils import setup_exp, plot_ims
from nets import create_ebm

import argparse

# random seed
tf.random.set_seed(1234)


def calc_fid(act1, act2):
  mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
  mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
  # calculate sum squared difference between means
  ssdiff = np.sum((mu1 - mu2)**2.0)
  # calculate sqrt of product between cov
  covmean = sqrtm(sigma1.dot(sigma2))
  # check and correct imaginary numbers from sqrt
  if np.iscomplexobj(covmean):
    covmean = covmean.real
  # calculate score
  fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
  return fid

def run_fid(strategy, config, inception, ebm, gen=None, train_iterator=None, save_str='samples.pdf'):

  def langevin_update(images_in, num_steps, noise_factor=1):
    # energy record and grad record
    energy_record = tf.zeros([num_steps, images_in.shape[0]])
    grad_record = tf.zeros([num_steps, images_in.shape[0]])

    images_samp = tf.identity(images_in)
    if config['mcmc_init'] == 'coop' and noise_factor != 0:
      # re-draw samples to avoid duplication on tpu device
      images_samp = tf.identity(gen(images_samp))
    # save init states
    images_init = tf.identity(images_samp)

    # langevin updates
    if num_steps > 0:
      for i in tf.range(int(num_steps)):
        with tf.GradientTape() as tape:
          tape.watch(images_samp)
          energy = ebm(images_samp, training=False) / config['mcmc_temp']
          # prior energy and scaling of joint energy
          if prior_ebm is not None:
            energy += prior_ebm(images_samp, training=False) / config['prior_temp']
            energy /= config['joint_temp']
          if 'tau' in config.keys() and config['tau'] > 0:
            energy += tf.reshape(tf.math.reduce_sum(tf.reshape(images_samp**2, (per_replica_batch_size, -1)), axis=1) / (2 * config['tau']**2), (-1, 1))
          energy_sum = tf.math.reduce_sum(energy)
        grads = tape.gradient(energy_sum, images_samp)
        # clip gradient norm (set to large value that won't interfere with standard dynamics)
        if config['clip_langevin_grad']:
          grads = tf.clip_by_norm(grads, config['max_langevin_norm'] / ((config['epsilon'] ** 2) / 2), axes=[1, 2, 3])

        # update images
        images_samp -= ((config['epsilon'] ** 2) / 2) * grads
        images_samp += noise_factor * config['epsilon'] * tf.random.normal(shape=tpu_tensor_size)

        # update energy record
        energy_record = tf.tensor_scatter_nd_update(energy_record, tf.reshape(i, [1, 1]), tf.reshape(energy, [1, -1]))
        # record gradient norm
        grad_norm = tf.norm(tf.reshape(grads, shape=[images_in.shape[0], -1]), axis=1)
        grad_norm *= ((config['epsilon'] ** 2) / 2)
        grad_record = tf.tensor_scatter_nd_update(grad_record, tf.reshape(i, [1, 1]), tf.reshape(grad_norm, [1, -1]))

    return images_samp, images_init, energy_record, grad_record

  @tf.function
  def resize_and_predict(images):
    images = tf.clip_by_value(images, clip_value_min=-1, clip_value_max=1)
    images = tf.image.resize(images, (299,299))
    return inception(images)

  # set up tf graphs for sampling and gradient updates
  @tf.function
  def langevin_mcmc(x):
    return langevin_update(x, config['mcmc_steps'], 1)

  if 'grad_steps' in config.keys() and config['grad_steps'] > 0:
    @tf.function
    def langevin_grad(x):
      return langevin_update(x, config['grad_steps'], 0)

  # get per replica batch size and set up containers for activation results
  per_replica_batch_size = config['batch_size'] // strategy.num_replicas_in_sync

  activations_1 = np.zeros((0, 2048))
  activations_2 = np.zeros((0, 2048))

  # loop over batches to get samples and calculate activations
  for i in range(config['num_fid_rounds']):
    print('Batch {} of {}'.format(i+1, config['num_fid_rounds']))

    # data images
    images_data = next(train_iterator)

    # get initial states to begin sampling
    if config['mcmc_init'] == 'data':
      sample_init = next(train_iterator)
    elif config['mcmc_init'] == 'coop':
      #z_init_tf = tf.random.normal([config['batch_size'], config['z_sz']])
      z_init_tf = gen.generate_latent_z(config['batch_size'])
      def get_z_init(ctx):
        rep_id = ctx.replica_id_in_sync_group
        return z_init_tf[(rep_id*per_replica_batch_size):((rep_id+1)*per_replica_batch_size)]
      sample_init = strategy.experimental_distribute_values_from_function(get_z_init)
    else:
      raise ValueError('Invalid mcmc_init')

    # mcmc sampling
    images_sample, images_init, energy_rec, grad_rec = strategy.run(langevin_mcmc, args=(sample_init,))

    if 'grad_steps' in config.keys() and config['grad_steps'] > 0:
      # perform "denoising" step with gradient only mcmc sampling
      images_sample, _, _, _ = strategy.run(langevin_grad, args=(images_sample,))

    # visualize batch of synthesized images and energy record
    if i == 0:
      # plot sampled images
      plot_ims(os.path.join(config['exp_dir'], config['exp_name'], 'images/init_' + save_str), 
               strategy.gather(images_init, 0))
      np.save(os.path.join(config['exp_dir'], config['exp_name'], 'numpy/init_samples.npy'), strategy.gather(images_init, 0).numpy())
      # plot sampled images
      plot_ims(os.path.join(config['exp_dir'], config['exp_name'], 'images/' + save_str), 
               strategy.gather(images_sample, 0))
      np.save(os.path.join(config['exp_dir'], config['exp_name'], 'numpy/samples.npy'), strategy.gather(images_sample, 0))
      # plot energy path
      plt.plot(strategy.gather(energy_rec, 1).numpy())
      plt.savefig(os.path.join(config['exp_dir'], config['exp_name'], 'plots/en_' + save_str))
      plt.close()
      # plot gradient path
      plt.plot(strategy.gather(grad_rec, 1).numpy())
      plt.savefig(os.path.join(config['exp_dir'], config['exp_name'], 'plots/grad_' + save_str))
      plt.close()

    # get activations from inception network
    act_images  = strategy.run(resize_and_predict, args=(images_data,))
    act_samples = strategy.run(resize_and_predict, args=(images_sample,))
    # gather from tpu and convert to numpy
    act_images = strategy.gather(act_images, 0).numpy()
    act_samples = strategy.gather(act_samples, 0).numpy()
    # save with other images
    activations_1 = np.concatenate((activations_1, act_images), 0)
    activations_2 = np.concatenate((activations_2, act_samples), 0)

  print('calculating FID score')
  fid_score = calc_fid(activations_1, activations_2)
  return fid_score


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('config_name', help='string for TPU device ID')
  args = parser.parse_args()

  ###############
  # ## SETUP ## #
  ###############

  # get experiment config
  config_module = importlib.import_module(args.config_name.replace('/', '.')[:-3])
  config = config_module.config

  # give exp_name unique timestamp identifier
  config['exp_name'] = config['exp_name'] + '_' + datetime.now().strftime('%y-%m-%d-%H-%M-%S')
  # manage experiment type (single ebm checkpoint by default, folder otherwise)
  if not 'exp_type' in config.keys():
    config['exp_type'] = 'single'
  assert config['exp_type'] in ['single', 'folder'], 'Only "single" or "folder" exp_type allowed.'

  # list of files to save
  code_file_list = ['fid.py', 'nets.py', 'utils.py', 'data.py', 'init.py', args.config_name]
  # setup folders, save code, set seed and get device
  setup_exp(os.path.join(config['exp_dir'], config['exp_name']), 
            ['images', 'plots', 'numpy'], 
            [os.path.join(config['root_path'], code_file) for code_file in code_file_list],
            config['gs_path'])

  # initialized distribution strategy
  strategy = init_strategy(config)


  ##################################################
  # ## INITIALIZE NETS, DATA, PERSISTENT STATES ## #
  ##################################################
  
  if 'prior_weights' in config.keys() and config['prior_weights'] is not None:
    prior_ebm = create_ebm(config)
    prior_ebm.load_weights(config['prior_weights'])
  else:
    prior_ebm = None

  if config['exp_type'] == 'single':
    # load nets and optim
    ebm, _, gen, _ = initialize_nets_and_optim(config, strategy)
    ebm.trainable = False
    if gen is not None:
      gen.trainable = False

    # test deterministic ouput
    determinism_test(config, strategy, ebm, gen)

    # load data
    train_iterator, _, _ = initialize_data(config, strategy)

    # Calculate per replica batch size, and distribute the datasets
    per_replica_batch_size = config['batch_size'] // strategy.num_replicas_in_sync
    batch_size = per_replica_batch_size * strategy.num_replicas_in_sync
    tpu_tensor_size = [per_replica_batch_size] + config['image_dims']

    with strategy.scope():
      inception = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
      inception.trainable=False

    fid_val = run_fid(strategy, config, inception, ebm, gen, train_iterator,
                      save_str='samples.pdf')

    # print and save results
    print('fid score: ', fid_val)
    out_file = open(os.path.join(config['exp_dir'], config['exp_name'], 'fid_rec.txt'), 'a')
    out_file.write(str(fid_val)+'\n')
    out_file.close()

  elif config['exp_type'] == 'folder':

    # get list of checkpoints to use 
    ckpt_step_list = np.arange(config['min_step'], config['max_step']+config['step_freq'], 
                                config['step_freq'])
    ckpt_strs = [str(step) + '.ckpt' for step in ckpt_step_list]
    # numpy container for exp results
    fid_vals = np.zeros(len(ckpt_strs))

    for i, ckpt_str in enumerate(ckpt_strs):
      # reset config for ebm weights (and gen weights if using coop init)
      config['ebm_weights'] = os.path.join(config['ckpt_folder'], 'ebm_'+ckpt_str)
      if config['mcmc_init'] == 'coop' and not config['fixed_gen']:
        config['gen_weights'] = os.path.join(config['ckpt_folder'], 'gen_'+ckpt_str)

      # load nets and optim
      ebm, _, gen, _ = initialize_nets_and_optim(config, strategy)
      ebm.trainable = False
      if gen is not None:
        gen.trainable = False
        # test initial samples for troubleshooting
        plot_ims(os.path.join(config['exp_dir'], config['exp_name'], 'images/test.png'), 
                  strategy.gather(gen.generate_images(config['batch_size']), 0))
      # test deterministic ouput
      determinism_test(config, strategy, ebm, gen)

      # load data
      train_iterator, _, _ = initialize_data(config, strategy)

      # Calculate per replica batch size, and distribute the datasets
      per_replica_batch_size = config['batch_size'] // strategy.num_replicas_in_sync
      batch_size = per_replica_batch_size * strategy.num_replicas_in_sync
      tpu_tensor_size = [per_replica_batch_size] + config['image_dims']

      with strategy.scope():
        inception = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
        inception.trainable=False

      # get fid score for single exp
      fid_val = run_fid(strategy, config, inception, ebm, gen, train_iterator, 
                        save_str='samples_' + str(ckpt_step_list[i]) + '.pdf')

      # update record
      fid_vals[i] = fid_val

      # print and save results
      print('Exp {} fid score: {}'.format(ckpt_str, fid_val))

      # save fid scores
      np.save(os.path.join(config['exp_dir'], config['exp_name'], 'fid_vals.npy'), fid_vals[0:(i+1)])
      # plot fid vs. step
      plt.plot(ckpt_step_list[0:(i+1)], fid_vals[0:(i+1)])
      plt.savefig(os.path.join(config['exp_dir'], config['exp_name'], 'fid_plot.pdf'), format='pdf')
      plt.close()

    print('Fid calculations concluded.')
