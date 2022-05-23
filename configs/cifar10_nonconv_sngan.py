####################
# ## PARAMETERS ## #
####################

config = {
  # paths for connecting to cloud storage
  "root_path": 'ebm-life-cycle/',
  "gs_path": None,

  # device type ('tpu' or 'gpu' or 'cpu')
  "device_type": 'gpu',

  # experiment info
  "exp_name": 'nonconv_sngan',
  "exp_dir": 'tf_out/cifar10',
  "num_training_steps": 150000,
  "batch_size": 128,
  "image_dims": [32, 32, 3],

  # data type and augmentation parameters
  "data_type": 'cifar10',
  "data_dir": 'cifar10_tfds',
  "data_download": True,
  "data_epsilon": 1e-2,

  # ebm network
  "net_type": 'ebm_sngan',

  # learning rate for ebm
  "ebm_optim_type": 'adam',
  "ebm_lr_info": [[1e-4, 0]], # [lr, step#]
  "ebm_opt_decay": 0.9999,

  # langevin sampling parameters
  "epsilon": 5e-3,
  "mcmc_step_info": [[100, 0, 0.5, 0]], # [num_mcmc_steps (K), num_burnin_steps (0), rejuvenation prob., step#]
  "mcmc_init": "coop_persistent",
  "mcmc_temp": 1e-4,
  "max_mcmc_updates": 2,

  # persistent image bank parameters
  "persistent_size": 10000,

  # generator network parameters
  "gen_type": "gen_sngan",
  "z_sz": 128,

  # learning rate for gen
  "gen_optim_type": 'adam',
  "gen_lr_info": [[1e-4, 0]],
  "gen_opt_decay": 0.9999,
  "update_generator": True,
  "gen_batch_norm": True,

  # gradient clipping
  "max_grad_norm": 20.0,
  "clip_ebm_grad": False,
  "clip_gen_grad": False,
  "max_langevin_norm": 0.1,
  "clip_langevin_grad": False,

  # logging parameters
  "info_freq": 500,
  "log_freq": 5000,
  "save_networks": True
}
