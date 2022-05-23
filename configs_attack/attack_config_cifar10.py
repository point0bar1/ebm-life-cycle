####################
# ## PARAMETERS ## #
####################

config={
  # paths for connecting to cloud storage
  "root_path": 'ebm-life-cycle/',
  "gs_path": None,

  # device type ('tpu' or 'gpu' or 'cpu')
  "device_type": 'gpu',

  "data_type": 'cifar10',
  "data_dir": 'cifar10_tfds',
  "data_download": True,
  'split': 'test',

  'exp_dir': 'tf_out/attack',
  'exp_name': 'cifar10_test',
  'image_dims': [32, 32, 3],

  'image_dims_clf': [32, 32, 3],
  'pixel_scale_clf': [-1, 1],

  'start_batch': 1,
  'end_batch': 10,
  'batch_size': 64,
  'shuffle': True,

  'adv_steps': 50,
  'adv_eps': 8.0,
  'adv_eta': 2.0,
  'adv_rand_start': True,

  'eot_attack_reps': 48,
  'eot_defense_reps': 128,

  'langevin_steps': 2000,
  'epsilon': 0.01,
  'mcmc_temp': 1e-4,

  'ebm_weights': "", # put path to ebm weights here
  'net_type': 'ebm_sngan',

  'clf_weights': "", # put path to classifier weights here

  'log_freq': 1,
  'record_image_states': False
}
