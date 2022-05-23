####################
# ## PARAMETERS ## #
####################

config = {
  # paths for connecting to cloud storage
  "root_path": 'ebm-life-cycle/',
  "gs_path": None,
  "exp_name": 'celeb_a_longrun_test',
  "exp_dir": 'tf_out/fid',

  # data type and augmentation parameters
  "data_type": 'celeb_a',
  "data_dir": 'celeb_a_tfds',
  "data_download": False,

  # device type ('tpu' or 'gpu' or 'cpu')
  "device_type": 'gpu',

  # exp params
  "exp_type": "folder",
  "num_fid_rounds": 53,
  "batch_size": 96,
  "image_dims": [64, 64, 3],
  "split": "train",
 
  # ckpt folder (only used if exp_type="folder")
  "ckpt_folder": '', # folder for trained checkpoint
  # min, max and frequency of steps to check
  "step_freq": 5000,
  "min_step": 140000,
  "max_step": 140000,
 
  # ebm network
  "net_type": 'ebm_sngan',

  # prior paramters
  "prior_weights": "", # once prior EBM is trained, put weight path here
  "prior_temp": 1e-4,

  # langevin sampling parameters
  "mcmc_steps": 1000000,
  "epsilon": 1e-2,
  "mcmc_init": "coop",
  "mcmc_temp": 1e-4,
  "tau": 1.5e-1,
  "joint_temp": 1.01,
  # clipping parameters
  "clip_langevin_grad": False,
  "max_langevin_norm": 100,

  "gen_type": "gen_sngan",
  "z_sz": 128,
  "fixed_gen": True,
  "gen_weights": "", # path to weights for generator (used for mcmc initialization)
}
