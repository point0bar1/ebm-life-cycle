###############################
# ## WIDERESNET CLASSIFIER ## #
###############################

config={
  # paths for connecting to cloud storage
  "root_path": 'ebm-life-cycle/',
  "gs_path": None,

  # device type ('tpu' or 'gpu' or 'cpu')
  "device_type": 'gpu',

  'data_type': 'cifar10',
  "data_dir": 'cifar10_tfds',
  "data_download": True,

  'exp_dir': 'tf_out/clf',
  'exp_name': 'cifar10_wrn',
  'image_dims': [32, 32, 3],
  'random_crop': True,

  'batch_size': 96,
  'num_epochs': 100,
  'epoch_steps_tr': 499,
  'epoch_steps_test': 99,

  'lr_list': [0.1, 0.01, 0.001],
  'lr_switch_epochs': [40, 60],
  'weight_decay': 0.0005,
  'momentum': 0.9,
  'nesterov': False,
  'dropout': 0.0,

  'test_and_log_freq': 5,
  "save_networks": True
}
