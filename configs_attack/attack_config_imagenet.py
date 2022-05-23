####################
# ## PARAMETERS ## #
####################

config={
  # paths for connecting to cloud storage
  "root_path": 'ebm-life-cycle/',
  "gs_path": None,

  # device type ('tpu' or 'gpu' or 'cpu')
  "device_type": 'gpu',

  'data_type': 'imagenet2012',
  "data_dir": 'imagenet2012_tfds',
  "data_download": False,
  'split': 'validation',

  'exp_dir': 'tf_out/attack',
  'exp_name': 'efficientnet_test',
  'image_dims': [224, 224, 3],

  'image_dims_clf': [600, 600, 3],
  'pixel_scale_clf': [0, 255],

  'start_batch': 1,
  'end_batch': 100,
  'batch_size': 8,
  'shuffle': True,

  'adv_steps': 50,
  'adv_eps': 2.0,
  'adv_eta': 1.0,
  'adv_rand_start': True,

  'eot_attack_reps': 24,
  'eot_defense_reps': 64,

  'langevin_steps': 200,
  'epsilon': 2e-2,
  'mcmc_temp': 1e-5,

  'ebm_weights': "", # put path to ebm weights here
  'net_type': 'ebm_biggan',

  'log_freq': 1,
  'record_image_states': False
}
