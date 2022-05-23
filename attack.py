import os
from time import time
from datetime import datetime
import importlib

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.efficientnet import EfficientNetB7

from utils import setup_exp
from init import init_strategy, initialize_data, determinism_test
from nets import create_ebm, WideResNetTF2

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
# rescale adv_eps and adv_eta to the correct range
config['adv_eps'] *= 2.0 / 255.0
config['adv_eta'] *= 2.0 / 255.0

# give exp_name unique timestamp identifier
time_str = datetime.now().strftime('%y-%m-%d-%H-%M-%S')
config['exp_name'] = config['exp_name'] + '_' + time_str

# setup folders, save code, set seed and get device
setup_exp(os.path.join(config['exp_dir'], config['exp_name']), 
          ['data_log'], 
          [os.path.join(config['root_path'], code_file) for code_file in
          ['attack.py', 'nets.py', 'utils.py', 'data.py', 'init.py', args.config_name]],
          config['gs_path'])

# initialize distribution strategy
strategy = init_strategy(config)


##################################################
# ## INITIALIZE NETS, DATA, PERSISTENT STATES ## #
##################################################

with strategy.scope():
  # Load imagenet classifier.
  if config['data_type'] == 'imagenet2012':
    # clf = MobileNetV2(include_top=True, weights='imagenet', input_shape=(224, 224, 3))
    clf = EfficientNetB7(include_top=True, weights='imagenet', input_shape=(600, 600, 3))
  else:
    clf = WideResNetTF2()
    clf.load_weights(config['clf_weights'])
  clf.trainable = False
  # load ebm
  ebm = create_ebm(config)
  ebm.load_weights(config['ebm_weights'])
  ebm.trainable = False
  # loss criterion for PGD (cross entropy)
  criterion = SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

# test deterministic ouput of ebm
determinism_test(config, strategy, ebm, None)
# test deterministic output of clf
with strategy.scope():
  z_test = tf.random.normal(shape=[3]+config['image_dims_clf'])
  z_out_1 = clf(z_test)
  z_out_2 = clf(z_test[0:2])
z_out_1 = strategy.gather(z_out_1, axis=0)
z_out_2 = strategy.gather(z_out_2, axis=0)
print('Classifier Determinism Test (if deterministic, should be very close to 0): ', 
      tf.math.reduce_max(tf.math.abs(z_out_1[0] - z_out_2[0])))


# Calculate per replica batch size, and distribute the datasets
print('Importing data...')
per_replica_batch_size = config['batch_size'] // strategy.num_replicas_in_sync
batch_size = per_replica_batch_size * strategy.num_replicas_in_sync
eot_ave_size = [per_replica_batch_size, config['eot_attack_reps']] + config['image_dims']
# batch size per replica for verification phase
verify_reps_per_replica = max(1, config['eot_defense_reps'] // strategy.num_replicas_in_sync)
eot_defense_reps = verify_reps_per_replica * strategy.num_replicas_in_sync
if not eot_defense_reps == config['eot_defense_reps']:
  print('Changed EOT defense reps from {} to {} for distribution across {} devices'.
        format(config['eot_defense_reps'], eot_defense_reps, strategy.num_replicas_in_sync))

# initialize data
attack_loader, _, _ = initialize_data(config, strategy, None, shuffle=config['shuffle'], repeat=False, get_label=True)


#################################################
# ## ATTACK AND DEFENSE FUNCTIONS FOR DEVICE ## #
#################################################

# function to scale ebm output in [-1, 1] range to range of other classifiers
def resize_and_rescale(X):
  # resize
  X_resized = X if config['image_dims']==config['image_dims_clf'] else tf.image.resize(X, config['image_dims_clf'][0:2])
  # scale pixel range
  if config['pixel_scale_clf'] == [-1, 1]:
    return X_resized
  elif config['pixel_scale_clf'] == [0, 1]:
    return (X_resized + 1.0) / 2.0
  elif config['pixel_scale_clf'] == [0, 255]:
    return 255.0 * (X_resized + 1.0) / 2.0
  else:
    raise ValueError('Invalid config option for pixel_scale_clf')

def purify(X, langevin_steps=0):
  X_purified = tf.identity(X)

  # langevin updates
  for i in tf.range(langevin_steps):
    with tf.GradientTape() as tape:
      tape.watch(X_purified)
      energy = tf.math.reduce_sum(ebm(X_purified, training=False)) / config['mcmc_temp']
    grads = tape.gradient(energy, X_purified)
    X_purified -= ((config['epsilon'] ** 2) / 2) * grads
    X_purified += config['epsilon'] * tf.random.normal(shape=(X_purified.shape))

  return X_purified

def eot_attack_loss(logits, y, reps=1):
  # finite-sample approximation of stochastic classifier loss for different EOT attack averaging methods
  # for deterministic logits with reps=1 this is just standard cross-entropy loss
  logits_loss = tf.reshape(logits, [int(logits.shape[0]/reps), reps, logits.shape[1]])
  logits_loss = tf.math.reduce_mean(logits_loss, axis=1)
  # final cross-entropy loss to generate attack grad
  loss = criterion(y, logits_loss)
  return loss

# TODO: replace with L_p for general p
def pgd_update_l_p(X_adv, grad, X):
  X_adv = X_adv + config['adv_eta'] * tf.sign(grad)
  X_adv = tf.math.minimum(X + config['adv_eps'], tf.math.maximum(X - config['adv_eps'], X_adv))
  X_adv = tf.clip_by_value(X_adv, -1.0, 1.0)
  return X_adv

@tf.function
def purify_and_predict_attack(X, y, X_orig):
  # replicate states
  X_repeat = tf.repeat(X, repeats=config['eot_attack_reps'], axis=0)
  # parallel purification of replicated states
  X_repeat_purified = purify(X_repeat, config['langevin_steps'])

  # return logits and loss (requires gradient on purified states for bpda)
  # predict labels of purified states
  with tf.GradientTape() as tape:
    tape.watch(X_repeat_purified)
    # resize image and change pixel range to match dims/range expected by clf
    X_purified_rescaled = resize_and_rescale(X_repeat_purified)
    # get logits and eot attack loss
    logits = clf(X_purified_rescaled, training=False)
    loss = eot_attack_loss(logits, y, config['eot_attack_reps'])
  # get BPDA gradients with respect to purified states
  X_grads = tape.gradient(loss, X_repeat_purified)
  attack_grads = tf.math.reduce_mean(tf.reshape(X_grads, eot_ave_size), axis=1)
  # update input state with attack gradient using PGD
  X_new = pgd_update_l_p(X, attack_grads, X_orig)

  return logits, X_new

@tf.function
def purify_and_predict_verify(X):
  # replicate states
  X_repeat = tf.repeat(X, repeats=verify_reps_per_replica, axis=0)
  # parallel purification of replicated states
  X_repeat_purified = purify(X_repeat, config['langevin_steps'])
  # resize image and change pixel range to match dims/range expected by clf
  X_purified_rescaled = resize_and_rescale(X_repeat_purified)
  # get logits and eot attack loss
  logits = clf(X_purified_rescaled, training=False)
  return logits

# TODO: replace with L_p for general p
@tf.function
def rand_init_l_p(X_adv):
  # random initialization in l_inf ball
  X_adv = tf.clip_by_value(X_adv + config['adv_eps'] * (2 * tf.random.uniform(X_adv.shape) - 1), -1, 1)
  return X_adv


#########################################
# ## ATTACK AND DEFENSE CONTROL FLOW ## #
#########################################

# this function will be used on CPU (not on device)
def eot_prediction(logits, y, reps=1):
  # finite-sample approximation of stochastic classifier for EOT defense averaging different methods
  # for deterministic logits with reps=1, this is just standard prediction
  if reps == 1:
    logits_pred = logits
  else:
    logits_pred = tf.reshape(logits, [int(logits.shape[0]/reps), reps, logits.shape[1]])
    logits_pred = tf.math.reduce_mean(logits_pred, axis=1)
  # finite sample approximation of stochastic classifier prediction
  y_pred = tf.math.argmax(logits_pred, axis=1)
  correct = tf.cast(y_pred == y, tf.float32)
  return correct

def eot_defense_verification(X_adv, y, correct, defended):
  # confirm that images are broken using a large sample size to evaluate the stochastic classifier
  for verify_ind in range(batch_size):
    if correct[verify_ind] == 0 and defended[verify_ind] == 1:
      # distribute states across devices
      X_single = tf.reshape(X_adv[verify_ind], [1] + config['image_dims'])
      def initialize_repeated_state(ctx):
        return X_single
      X_repeat = strategy.experimental_distribute_values_from_function(initialize_repeated_state)
      # run parallel mcmc on devices and get logits from clf
      logits_verify = strategy.run(purify_and_predict_verify, args=(X_repeat,))
      # get average logits and stochastic clf pred
      logits_verify_gather = strategy.gather(logits_verify, 0)
      verify_bool = eot_prediction(logits_verify_gather, y[verify_ind], eot_defense_reps)
      # update record
      defended = tf.tensor_scatter_nd_update(defended, tf.reshape(tf.constant(verify_ind), [1, 1]),
                                             tf.reshape(verify_bool, [1]))

  return defended

def eval_and_attack_step(X_adv, y, X_orig, X_adv_gather, y_gather, defended):
  # get logits for old adversarial state and updated adversarial state
  logits_adv, X_adv_new = strategy.run(purify_and_predict_attack, args=(X_adv, y, X_orig))
  # evalute logit predictions of old adv state and average across devices
  logits_adv_gather = strategy.gather(logits_adv, 0)
  correct = eot_prediction(logits_adv_gather, y_gather, config['eot_attack_reps'])
  # evaluate candidates for breaks using a large number of parallel MCMC samples
  defended = eot_defense_verification(X_adv_gather, y_gather, correct, defended)
  return defended, X_adv_new

def attack_batch(X, y, batch_num):
  # get baseline accuracy for natural images
  X_gather, y_gather = strategy.gather(X, 0), strategy.gather(y, 0)
  defended = eot_defense_verification(X_gather, y_gather, tf.zeros([batch_size]), tf.ones([batch_size]))
  print('Batch {} of {} Baseline: {} of {}'.
        format(batch - config['start_batch'] + 2, config['end_batch'] - config['start_batch'] + 1,
               int(tf.math.reduce_sum(defended).numpy()), batch_size))

  # record of defense over attacks
  class_batch = tf.zeros([config['adv_steps'] + 2, batch_size])
  class_batch = tf.tensor_scatter_nd_update(class_batch, tf.reshape(tf.constant(0), [1, 1]), 
                                            tf.reshape(defended, [1, -1]))
  # record for adversarial images for verified breaks
  adv_batch = tf.zeros([batch_size] + config['image_dims'])
  for ind in range(batch_size):
    if defended[ind] == 0:
      # record mis-classified natural images as adversarial states
      adv_batch = tf.tensor_scatter_nd_update(adv_batch, tf.reshape(tf.constant(ind), [1, 1]), 
                                              tf.expand_dims(X_gather[ind], 0))

  # start in random location of l_p ball
  if config['adv_rand_start']:
    X_adv = strategy.run(rand_init_l_p, args=(X,))

  # adversarial attacks on a single batch of images
  for step in range(config['adv_steps'] + 1):
    # initial state before update
    X_adv_gather = strategy.gather(X_adv, 0)

    # get attack gradient and update defense record
    defended, X_adv = eval_and_attack_step(X_adv, y, X, X_adv_gather, y_gather, defended)

    # update step-by-step defense record
    class_batch = tf.tensor_scatter_nd_update(class_batch, tf.reshape(tf.constant(step + 1), [1, 1]), 
                                              tf.reshape(defended, [1, -1]))
    # add adversarial images for newly broken images to list
    for ind in range(batch_size):
      if class_batch[step, ind] == 1 and defended[ind] == 0:
        adv_batch = tf.tensor_scatter_nd_update(adv_batch, tf.reshape(tf.constant(ind), [1, 1]), 
                                                tf.expand_dims(X_adv_gather[ind], 0))

    if step == 1 or step % config['log_freq'] == 0 or step == config['adv_steps']:
      # print attack info
      print('Batch {} of {}, Attack {} of {}   Batch defended: {} of {}'.
            format(batch_num - config['start_batch'] + 2, config['end_batch'] - config['start_batch'] + 1,
                    step, config['adv_steps'], int(tf.math.reduce_sum(defended).numpy()), batch_size))

  # record final adversarial image for unbroken states
  for ind in range(batch_size):
    if defended[ind] == 1:
      adv_batch = tf.tensor_scatter_nd_update(adv_batch, tf.reshape(tf.constant(ind), [1, 1]), 
                                              tf.expand_dims(X_adv_gather[ind], 0))

  return class_batch, adv_batch


########################################
# ## ATTACK CLASSIFIER AND PURIFIER ## #
########################################

# defense record for over attacks
class_path = np.zeros([config['adv_steps'] + 2, 0])
if config['record_image_states']:
  # record of original images, adversarial images, and labels
  labs = np.zeros([0], dtype=np.int64)
  ims_orig = np.zeros([0] + config['image_dims'])
  ims_adv = np.zeros([0] + config['image_dims'])

# run adversarial attacks on samples from image bank in small batches
print('\nAttack has begun.\n----------')
for batch, (X_batch, y_batch) in enumerate(attack_loader):
  if (batch + 1) < config['start_batch']:
    continue
  elif (batch + 1) > config['end_batch']:
    break
  else:
    if config['record_image_states']:
      # record original states and labels
      ims_orig = np.concatenate((ims_orig, strategy.gather(X_batch, 0).numpy()), 0)
      labs = np.concatenate((labs, strategy.gather(y_batch, 0).numpy()), 0)

    # attack images using setting in config
    class_batch, ims_adv_batch = attack_batch(X_batch, y_batch, batch)

    # update defense records
    class_path = np.concatenate((class_path, class_batch.numpy()), 1)
    if config['record_image_states']:
      # record adversarial images
      ims_adv = np.concatenate((ims_adv, ims_adv_batch.numpy()), 0)

    print('Attack concluded on Batch {} of {}. Total Secure Images: {} of {}\n-----------\n'.
          format(batch - config['start_batch'] + 2, config['end_batch'] - config['start_batch'] + 1,
                 int(tf.math.reduce_sum(class_path[config['adv_steps']+1, :]).numpy()), class_path.shape[1]))

    if config['record_image_states']:
      # save states and experiment profile
      np.savez(os.path.join(config['exp_dir'], config['exp_name'], 'data_log/results.npz'),
               ims_orig=ims_orig, ims_adv=ims_adv, labs=labs, class_path=class_path)
    else:
      # save experiment profile
      np.savez(os.path.join(config['exp_dir'], config['exp_name'], 'data_log/results.npz'), class_path=class_path)

    # final defense accuracy
    accuracy_baseline = float(tf.math.reduce_sum(class_path[0, :]).numpy()) / class_path.shape[1]
    accuracy_adv = float(tf.math.reduce_sum(class_path[config['adv_steps']+1, :]).numpy()) / class_path.shape[1]
    print('Attack Results for {} samples: Non-Adversarial {}    Adversarial: {}\n-----------\n'.
          format(class_path.shape[1], accuracy_baseline, accuracy_adv))

    # plot accuracy over attacks
    plt.plot(class_path.mean(1))
    plt.table(cellText=[[accuracy_baseline, accuracy_adv, class_path.shape[1]]],
              colLabels=['baseline', 'secure', 'total images'], bbox=[0.0, -0.35, 1, 0.125])
    plt.xlabel('attack')
    plt.ylabel('accuracy')
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(os.path.join(config['exp_dir'], config['exp_name'], 'data_log/accuracy_over_attack.png'))
    plt.close()
