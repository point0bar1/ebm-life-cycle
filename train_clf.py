import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

import os
from datetime import datetime
import importlib
import matplotlib.pyplot as plt
import numpy as np

from nets import WideResNetTF2, conv_init
from utils import setup_exp
from init import init_strategy, initialize_data, determinism_test

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
          ['checkpoints', 'plots'], 
          [os.path.join(config['root_path'], code_file) for code_file in
          ['train_clf.py', 'nets.py', 'utils.py', 'data.py', 'init.py', args.config_name]],
          config['gs_path'])

# initialize distribution strategy
strategy = init_strategy(config)


##################################
# ## INITIALIZE NETS AND DATA ## #
##################################

with strategy.scope():
  clf = WideResNetTF2(dropout=config['dropout'])

  # initialize optim
  lr_switch_epochs = [config['epoch_steps_tr'] * step for step in config['lr_switch_epochs']]
  lr_schedule = PiecewiseConstantDecay(lr_switch_epochs, config['lr_list'])
  optim = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=config['momentum'], nesterov=config['nesterov'])

  # loss criterion for PGD (cross entropy)
  criterion = SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

  # records
  training_loss = tf.keras.metrics.Mean('training_loss', dtype=tf.float32)
  training_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('training_accuracy', dtype=tf.float32)
  test_loss = tf.keras.metrics.Mean('testing_loss', dtype=tf.float32)
  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('testing_accuracy', dtype=tf.float32)

# test deterministic output of clf
determinism_test(config, strategy, clf, None)
# initialize clf weights
conv_init(clf)


######################################
# ## TF GRAPHS FOR TRAIN AND TEST ## #
######################################

@tf.function
def train_epoch(train_loader_in):
  def train_step(inputs):
    X, y = inputs
    with tf.GradientTape() as tape:
      logits = clf(X, training=True)
      loss = tf.math.reduce_mean(criterion(y, logits))
      if config['weight_decay'] > 0:
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in clf.trainable_variables])
        loss_total = loss + config['weight_decay'] * l2_loss
      else:
        loss_total = loss
      loss_total_rescaled = loss_total / strategy.num_replicas_in_sync
    clf_grads = tape.gradient(loss_total_rescaled, clf.trainable_variables)
    # update clf
    optim.apply_gradients(list(zip(clf_grads, clf.trainable_variables)))

    training_loss.update_state(loss)
    training_accuracy.update_state(y, logits)

  for _ in tf.range(config['epoch_steps_tr']):
    strategy.run(train_step, args=(next(train_loader_in),))

@tf.function
def test_epoch(test_loader_in):
  def test_step(inputs):
    X, y = inputs
    logits = clf(X, training=False)
    loss = tf.math.reduce_mean(criterion(y, logits))

    test_loss.update_state(loss)
    test_accuracy.update_state(y, logits)

  for _ in tf.range(config['epoch_steps_test']):
    strategy.run(test_step, args=(next(test_loader_in),))


#######################
# ## LEARNING LOOP # ##
#######################

# records for training info
train_loss_rec = np.zeros([config['num_epochs']])
train_acc_rec = np.zeros([config['num_epochs']])
test_loss_rec = np.zeros([config['num_epochs'] // config['test_and_log_freq']])
test_acc_rec = np.zeros([config['num_epochs'] // config['test_and_log_freq']])

print('Training has begun.')
for epoch in range(config['num_epochs']):
  clf.trainable = True
  config['split'] = 'train'
  train_loader, _, _ = initialize_data(config, strategy, None, shuffle=True, repeat=False, get_label=True)
  train_epoch(train_loader)
  print('Epoch {}: Training Loss={}   Training Acc={}%'.
          format(epoch+1, 
                 round(float(training_loss.result()), 4),
                 round(float(training_accuracy.result()) * 100, 2)))
  
  # update training record then reset the metric objects
  train_loss_rec[epoch] = round(float(training_loss.result()), 4)
  train_acc_rec[epoch] = round(float(training_accuracy.result()) * 100, 2)
  training_loss.reset_states()
  training_accuracy.reset_states()

  if (epoch+1) % config['test_and_log_freq'] == 0:
    # evaluate test data
    clf.trainable = False
    config['split'] = 'test'
    test_loader, _, _ = initialize_data(config, strategy, None, shuffle=False, repeat=False, get_label=True)
    test_epoch(test_loader)
    print('Epoch {}: Test Loss={}   Test Acc={}%'.
          format(epoch+1, 
                 round(float(test_loss.result()), 4),
                 round(float(test_accuracy.result()) * 100, 2)))

    # update training record then reset the metric objects
    test_loss_rec[epoch // config['test_and_log_freq']] = round(float(test_loss.result()), 4)
    test_acc_rec[epoch // config['test_and_log_freq']] = round(float(test_accuracy.result()) * 100, 2)
    test_loss.reset_states()
    test_accuracy.reset_states()

    # save checkpoint and diagnostic plots
    test_num = (epoch + 1) // config['test_and_log_freq']
    best_checkpoint = np.where(np.reshape(test_acc_rec[0:test_num], [test_num])==np.max(test_acc_rec[0:test_num]))[0][0]
    best_acc = test_acc_rec[best_checkpoint]
    best_checkpoint = (best_checkpoint + 1) * config['test_and_log_freq']

    plt.plot(np.arange(1, epoch+2), train_loss_rec[0:(epoch+1)])
    plt.plot(config['test_and_log_freq']*np.arange(1, test_num+1), test_loss_rec[0:test_num])
    plt.savefig(os.path.join(config['exp_dir'], config['exp_name'], 'plots', 'loss_fig.png'))
    plt.close()

    plt.plot(np.arange(1, epoch+2), train_acc_rec[0:(epoch+1)])
    plt.plot(config['test_and_log_freq']*np.arange(1, test_num+1), test_acc_rec[0:test_num])
    plt.table(cellText=[['Best Ckpt: ', str(best_checkpoint), ' Acc: ', str(best_acc)]], bbox=[0.0, -0.35, 1, 0.125])
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(os.path.join(config['exp_dir'], config['exp_name'], 'plots', 'acc_fig.png'))
    plt.close()

    # save network
    if config['save_networks']:

      if config['gs_path'] is not None:
        # name of google storage bucket
        gs_string = 'gs://' + config['gs_path']
      else:
        # empty string for local save
        gs_string = ''

      clf.save_weights(os.path.join(gs_string, config['exp_dir'], config['exp_name'], 
                                    'checkpoints/clf_{}.ckpt'.format(epoch+1)))
