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
  "exp_name": 'conv_sngan',
  "exp_dir": 'tf_out/celeb_a',
  "num_training_steps": 250000,
  "batch_size": 128,
  "image_dims": [64, 64, 3],

  # data type and augmentation parameters
  "data_type": 'celeb_a',
  "data_dir": 'celeb_a_tfds',
  "data_download": False,
  "data_epsilon": 2e-2,

  # ebm network
  "net_type": 'ebm_sngan',

  # learning rate for ebm
  "ebm_optim_type": 'adam',
  "ebm_lr_type": 'step',
  "ebm_lr_info": [[2.5e-5, 0], [1e-5, 50000], [1e-6, 75000], [1e-7, 100000], [1e-8, 125000]], # [lr, step#]
  "ebm_opt_decay": 0,

  # langevin sampling parameters
  "epsilon": 1e-2,
  "mcmc_step_info": [[100, 100, 750, 0]], # [num_mcmc_steps (K), burnin mcmc steps, update count threshold, step#]
  "mcmc_init": "coop_persistent",
  "mcmc_temp": 1e-4,
  "tau": 1.5e-1,

  # prior network parameters
  "prior_weights": "", # once prior EBM is trained, put weight path here
  "prior_temp": 1e-4,

  # persistent image bank parameters
  "persistent_size": 15000,
  "burnin_size": 1000,
  "burnin_grad_updates": 0,

  # generator network parameters
  "gen_type": "gen_sngan",
  "z_sz": 128,
  "update_generator": False,
  "gen_batch_norm": False,
  "gen_weights": "", # path to weights for generator (used for mcmc initialization)

  # learning rate for gen
  "gen_optim_type": 'adam',
  "gen_lr_info": [[1e-4, 0]],
  "gen_opt_decay": 0.9999,

  # gradient clipping
  "max_grad_norm": 20,
  "clip_ebm_grad": False,
  "max_langevin_norm": 1.0,
  "clip_langevin_grad": False,

  # logging parameters
  "info_freq": 500,
  "log_freq": 5000,
  "save_networks": True
}
