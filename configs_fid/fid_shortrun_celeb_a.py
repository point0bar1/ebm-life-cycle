####################
# ## PARAMETERS ## #
####################

config = {
  # paths for connecting to cloud storage
  "root_path": 'ebm-life-cycle/',
  "gs_path": None,
  "exp_name": 'celeb_a_shortrun_test',
  "exp_dir": 'tf_out/fid',

  # device type ('tpu' or 'gpu' or 'cpu')
  "device_type": 'gpu',

  # data type and augmentation parameters
  "data_type": 'celeb_a',
  "data_dir": 'celeb_a_tfds',
  "data_download": False,

  # exp params
  "exp_type": "folder",
  "num_fid_rounds": 520,
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

  # langevin sampling parameters
  "mcmc_steps": 300,
  "epsilon": 3e-3,
  "mcmc_init": "coop",
  "mcmc_temp": 1e-6,
  # denoising parameters
  "grad_steps": 0,
  # clipping parameters
  "clip_langevin_grad": False,
  "max_langevin_norm": 0.25,

  "gen_type": "gen_sngan",
  "z_sz": 128,
  "fixed_gen": False
}
