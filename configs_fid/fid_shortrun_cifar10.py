####################
# ## PARAMETERS ## #
####################

config = {
  # paths for connecting to cloud storage
  "root_path": 'ebm-life-cycle/',
  "gs_path": None,
  "exp_name": 'cifar10_shortrun_test',
  "exp_dir": 'tf_out/fid',

  # device type ('tpu' or 'gpu' or 'cpu')
  "device_type": 'gpu',

  # data type and augmentation parameters
  "data_type": 'cifar10',
  "data_dir": 'cifar10_tfds',
  "data_download": True,

  # exp params
  "exp_type": "single",
  "num_fid_rounds": 520,
  "batch_size": 96,
  "image_dims": [32, 32, 3],
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
  "mcmc_steps": 350,
  "epsilon": 5e-3,
  "mcmc_init": "coop",
  "mcmc_temp": 1e-4,
  # denoising parameters
  "grad_steps": 0,
  # clipping parameters
  "clip_langevin_grad": False,
  "max_langevin_norm": 0.1,

  "gen_type": "gen_sngan",
  "z_sz": 128,
  "fixed_gen": False
}
