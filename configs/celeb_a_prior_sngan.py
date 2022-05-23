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
  "exp_name": 'prior_sngan',
  "exp_dir": 'tf_out/celeb_a',
  "num_training_steps": 150000,
  "batch_size": 128,
  "image_dims": [64, 64, 3],

  # data type and augmentation parameters
  "data_type": 'celeb_a',
  "data_dir": 'celeb-a_tfds',
  "data_download": False,
  "data_epsilon": 2e-2,

  # ebm network
  "net_type": 'ebm_sngan',

  # learning rate for ebm
  "ebm_optim_type": 'adam',
  "ebm_lr_info": [[1e-4, 0]], # [lr, step#]
  "ebm_opt_decay": 0.9999,

  # langevin sampling parameters
  "epsilon": 1e-2,
  "mcmc_step_info": [[100, 0, 0.2, 0]], # [num_mcmc_steps (K), num_burnin_steps (0), rejuvenation prob., step#]
  "mcmc_init": "coop_persistent",
  "mcmc_temp": 1e-4,
  "tau": 1.5e-1,
  "max_mcmc_updates": 0,

  # persistent image bank parameters
  "persistent_size": 10000,

  # generator network parameters
  "gen_type": "gen_sngan",
  "z_sz": 128,
  "gen_weights": "", # path to weights for generator (used for mcmc initialization)

  # learning rate for gen
  # TODO: better LR schedule?
  "gen_optim_type": 'adam',
  "gen_lr_info": [[1e-4, 0]],
  "gen_opt_decay": 0.9999,
  "update_generator": False,
  "gen_batch_norm": False,

  # gradient clipping
  "max_grad_norm": 20.0,
  "clip_ebm_grad": False,
  "clip_gen_grad": False,
  "max_langevin_norm": 0.7,
  "clip_langevin_grad": False,

  # logging parameters
  "info_freq": 500,
  "log_freq": 5000,
  "save_networks": True
}
