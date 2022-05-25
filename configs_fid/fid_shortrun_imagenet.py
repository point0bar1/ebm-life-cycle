####################
# ## PARAMETERS ## #
####################

config = {
  # paths for connecting to cloud storage
  "root_path": 'ebm-life-cycle/',
  "gs_path": None,
  "exp_name": 'imagenet_shortrun_test',
  "exp_dir": 'tf_out/fid',

  # device type ('tpu' or 'gpu' or 'cpu')
  "device_type": 'gpu',

  # data type and augmentation parameters
  "data_type": 'imagenet2012',
  "data_dir": 'imagenet2012_tfds',
  "data_download": False,
  "random_crop": False,

  # exp params
  "exp_type": "single",
  "num_fid_rounds": 520,
  "batch_size": 96,
  "image_dims": [128, 128, 3],
  "split": "train",

  # ebm and gen files (only used if exp_type="single")
  "ebm_weights": "",
  "gen_weights": "",

  # ckpt folder (only used if exp_type="folder")
  "ckpt_folder": '', # folder for trained checkpoint
  # min, max and frequency of steps to check
  "step_freq": 5000,
  "min_step": 140000,
  "max_step": 140000,
 
  # ebm network
  "net_type": 'ebm_sngan',

  # langevin sampling parameters
  "mcmc_steps": 320,
  "epsilon": 3e-3,
  "mcmc_init": "coop",
  "mcmc_temp": 1e-7,
  # clipping parameters
  "max_langevin_norm": 2.0,
  "clip_langevin_grad": False,

  "gen_type": "gen_sngan",
  "z_sz": 128,
  "fixed_gen": False
}
