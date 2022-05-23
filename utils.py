import os
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from google.cloud import storage


def upload_blob(bucket_name, source_file_name, destination_blob_name):
  """Uploads a file to the bucket."""

  storage_client = storage.Client()
  bucket = storage_client.bucket(bucket_name)
  blob = bucket.blob(destination_blob_name)

  blob.upload_from_filename(source_file_name)

  print(
    "File {} uploaded to {}.".format(
      source_file_name, destination_blob_name
    )
  )

def download_blob(bucket_name, source_file_name, destination_blob_name):
  """Downloads a file from the bucket."""

  storage_client = storage.Client()
  bucket = storage_client.get_bucket(bucket_name)
  blob = bucket.blob(source_file_name)

  blob.download_to_filename(destination_blob_name)

  print(
    "File {} downloaded to {}.".format(
      source_file_name, destination_blob_name
    )
  )

# visualize images with pixels in range [-1, 1]
def plot_ims(path, ims): 
  ims = (np.clip(ims.numpy(), -1., 1.) + 1) / 2
  fig = plt.figure(figsize=(int(ims.shape[0] ** 0.5), int(ims.shape[0] ** 0.5)))
  grid = ImageGrid(
    fig, 111,  # similar to subplot(111)
    nrows_ncols=(int(ims.shape[0] ** 0.5), int(ims.shape[0] ** 0.5)),
    axes_pad=0.05,  # pad between axes in inch.
  )
  
  grid[0].get_yaxis().set_ticks([])
  grid[0].get_xaxis().set_ticks([])

  for ax, im in zip(grid, ims.tolist()):
    im = np.array(im)
    if im.shape[2] == 1:
      im = np.tile(im, (1, 1, 3))
    ax.imshow(im)
    ax.axis("off")
  plt.savefig(path, format="pdf", dpi=2000)
  plt.close()

# save copy of code in the experiment folder
def save_code(exp_dir, code_file_list, gs_path=None, save_to_cloud=True):
  def save_file(file_name):
    file_in = open(file_name, 'r')
    file_out = open(os.path.join(exp_dir, 'code/', os.path.basename(file_name)), 'w')
    for line in file_in:
      file_out.write(line)
  for code_file in code_file_list:
    save_file(code_file)
    if gs_path is not None and save_to_cloud == True:
      upload_blob(gs_path,
                  os.path.join(exp_dir, 'code/', os.path.basename(code_file)), 
                  os.path.join(exp_dir, 'code/', os.path.basename(code_file))
      )

# make folders, save config and code, get device, set seed
def setup_exp(exp_dir, folder_list, code_file_list=[], gs_path=None, save_to_cloud=True):
  # make directory for saving results
  if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
  for folder in ['code'] + folder_list:
    if not os.path.exists(os.path.join(exp_dir, folder)):
      os.mkdir(os.path.join(exp_dir, folder))
  save_code(exp_dir, code_file_list, gs_path, save_to_cloud)

# plot diagnostics for learning
def plot_diagnostics(config, step, loss_rec, grad_norm_rec, fontsize=6):
  exp_folder = os.path.join(config['exp_dir'], config['exp_name'])

  loss_rec = loss_rec.numpy()
  grad_norm_rec = grad_norm_rec.numpy()

  # save numpy files of loss and grad norm
  np.save(os.path.join(exp_folder, 'plots/loss_rec.npy'), loss_rec)
  np.save(os.path.join(exp_folder, 'plots/grad_rec.npy'), grad_norm_rec)
  if config['save_networks'] and config['gs_path'] is not None:
    upload_blob(config['gs_path'],
                os.path.join(exp_folder, 'plots/loss_rec.npy'),
                os.path.join(exp_folder, 'plots/loss_rec.npy'))
    upload_blob(config['gs_path'],
                os.path.join(exp_folder, 'plots/grad_rec.npy'),
                os.path.join(exp_folder, 'plots/grad_rec.npy'))

  # axis tick size
  matplotlib.rc('xtick', labelsize=6)
  matplotlib.rc('ytick', labelsize=6)
  fig = plt.figure()

  def plot_en_diff_and_grad_mag():
    # energy difference
    ax = fig.add_subplot(221)
    ax.plot(loss_rec)
    ax.axhline(y=0, ls='--', c='k')
    ax.set_title('Energy Difference', fontsize=fontsize)
    ax.set_xlabel('batch', fontsize=fontsize)
    ax.set_ylabel('loss', fontsize=fontsize)
    # mean langevin gradient
    ax = fig.add_subplot(222)
    ax.plot(grad_norm_rec)
    ax.set_title('Average Langevin \n Gradient Magnitude', fontsize=fontsize)
    ax.set_xlabel('batch', fontsize=fontsize)
    ax.set_ylabel('grad norm', fontsize=fontsize)

  def plot_crosscorr_and_autocorr(t_gap_max=5000, max_lag=15, b_w=0.35):
    t_init = max(0, step + 1 - t_gap_max)
    t_end = step + 1
    t_gap = t_end - t_init
    max_lag = min(max_lag, t_gap - 1)
    # rescale energy diffs to unit mean square but leave uncentered
    loss_rescale = loss_rec[t_init:t_end] / np.sqrt(np.sum(loss_rec[t_init:t_end] * loss_rec[t_init:t_end])/(t_gap-1))
    # normalize gradient magnitudes
    grad_rescale = (grad_norm_rec[t_init:t_end]-np.mean(grad_norm_rec[t_init:t_end]))/np.std(grad_norm_rec[t_init:t_end])
    # cross-correlation and auto-correlations
    cross_corr = np.correlate(loss_rescale, grad_rescale, 'full') / (t_gap - 1)
    loss_acorr = np.correlate(loss_rescale, loss_rescale, 'full') / (t_gap - 1)
    grad_acorr = np.correlate(grad_rescale, grad_rescale, 'full') / (t_gap - 1)
    # x values and indices for plotting
    x_corr = np.linspace(-max_lag, max_lag, 2 * max_lag + 1)
    x_acorr = np.linspace(0, max_lag, max_lag + 1)
    t_0_corr = int((len(cross_corr) - 1) / 2 - max_lag)
    t_0_acorr = int((len(cross_corr) - 1) / 2)

    # plot cross-correlation
    ax = fig.add_subplot(223)
    ax.bar(x_corr, cross_corr[t_0_corr:(t_0_corr + 2 * max_lag + 1)])
    ax.axhline(y=0, ls='--', c='k')
    ax.set_title('Cross Correlation of Energy Difference\nand Gradient Magnitude', fontsize=fontsize)
    ax.set_xlabel('lag', fontsize=fontsize)
    ax.set_ylabel('correlation', fontsize=fontsize)
    # plot auto-correlation
    ax = fig.add_subplot(224)
    ax.bar(x_acorr-b_w/2, loss_acorr[t_0_acorr:(t_0_acorr + max_lag + 1)], b_w, label='loss')
    ax.bar(x_acorr+b_w/2, grad_acorr[t_0_acorr:(t_0_acorr + max_lag + 1)], b_w, label='grad. mag.')
    ax.axhline(y=0, ls='--', c='k')
    ax.set_title('Auto-Correlation of Energy Difference\nand Gradient Magnitude', fontsize=fontsize)
    ax.set_xlabel('lag', fontsize=fontsize)
    ax.set_ylabel('correlation', fontsize=fontsize)
    ax.legend(loc='upper right', fontsize=fontsize-4)

  # make diagnostic plots
  plot_en_diff_and_grad_mag()
  plot_crosscorr_and_autocorr()
  # save figure
  plt.subplots_adjust(hspace=0.6, wspace=0.6)
  plt.savefig(os.path.join(exp_folder, 'plots', 'diagnosis_plot.png'))
  plt.close()
  if config['save_networks'] and config['gs_path'] is not None:
    upload_blob(config['gs_path'],
                os.path.join(exp_folder, 'plots', 'diagnosis_plot.png'),
                os.path.join(exp_folder, 'plots', 'diagnosis_plot.png'))

# function for saving model results
def save_model(config, step, strategy, ebm, ebm_optim, ims_samp, ims_persistent=None, 
               z_persistent=None, gen=None, gen_optim=None):

  import pickle

  # folder for results
  exp_folder = os.path.join(config['exp_dir'], config['exp_name'])
  per_replica_batch_size = config['batch_size'] // strategy.num_replicas_in_sync

  # save example images
  images_samp_viz = strategy.gather(ims_samp, axis=0)
  plot_ims(os.path.join(exp_folder, 'shortrun/shortrun'+str(step+1)+'.pdf'), 
           images_samp_viz[0:per_replica_batch_size])
  if config['save_networks'] and config['gs_path'] is not None:
    upload_blob(config['gs_path'],
                os.path.join(exp_folder, 'shortrun/shortrun'+str(step+1)+'.pdf'),
                os.path.join(exp_folder, 'shortrun/shortrun'+str(step+1)+'.pdf'))

  # save persistent state samples
  if config['mcmc_init'].endswith('persistent'):
    persistent_states_gather = strategy.gather(ims_persistent, axis=0)
    plot_ims(os.path.join(exp_folder, 'shortrun/persistent'+str(step+1)+'.pdf'), 
              persistent_states_gather[0:per_replica_batch_size])
    if config['save_networks'] and config['gs_path'] is not None:
      upload_blob(config['gs_path'],
                  os.path.join(exp_folder, 'shortrun/persistent'+str(step+1)+'.pdf'),
                  os.path.join(exp_folder, 'shortrun/persistent'+str(step+1)+'.pdf'))

  # generator samples
  if config['mcmc_init'].startswith('coop'):
    # save entire sanity check path of states
    z_check = gen.generate_latent_z(num_ims=per_replica_batch_size)
    gen_check = gen(z_check, training=False)
    ebm_check = ebm(gen_check)
    # save images and numpy files of sanity check states
    #np.save(os.path.join(exp_folder, 'shortrun/z_check_'+str(step+1)+'.npy'), z_check.numpy())
    #np.save(os.path.join(exp_folder, 'shortrun/gen_check_'+str(step+1)+'.npy'), gen_check.numpy())
    #np.save(os.path.join(exp_folder, 'shortrun/ebm_check_'+str(step+1)+'.npy'), ebm_check.numpy())
    plot_ims(os.path.join(exp_folder, 'shortrun/generator'+str(step+1)+'.pdf'), gen_check)
    if config['save_networks'] and config['gs_path'] is not None:
      upload_blob(config['gs_path'],
                  os.path.join(exp_folder, 'shortrun/generator'+str(step+1)+'.pdf'),
                  os.path.join(exp_folder, 'shortrun/generator'+str(step+1)+'.pdf'))

  if config['save_networks'] and step > 0:

    if config['gs_path'] is not None:
      # name of google storage bucket
      gs_string = 'gs://' + config['gs_path']
    else:
      # empty string for local save
      gs_string = ''

    # save model
    ebm.save_weights(os.path.join(gs_string, exp_folder, 'checkpoints/ebm_{}.ckpt'.format(step+1)))
    # save optim
    #with open(os.path.join(exp_folder, 'checkpoints/ebm_optim_{}.ckpt'.format(step+1)), 'wb') as f:
    #  pickle.dump(ebm_optim.get_weights(), f)
    #upload_blob(config['gs_path'],
    #            os.path.join(exp_folder, 'checkpoints/ebm_optim_'+str(step+1)+'.ckpt'),
    #            os.path.join(exp_folder, 'checkpoints/ebm_optim_'+str(step+1)+'.ckpt'))

    # save persistent states to cloud
    #if ims_persistent is not None:
    #  with open(os.path.join(exp_folder, 'checkpoints/persistent.ckpt'), 'wb') as f:
    #    pickle.dump(persistent_states_gather, f)
    #  upload_blob(config['gs_path'],
    #              os.path.join(exp_folder, 'checkpoints/persistent.ckpt'),
    #              os.path.join(exp_folder, 'checkpoints/persistent.ckpt'))
    #  # remove to save space
    #  os.remove(os.path.join(exp_folder, 'checkpoints/persistent.ckpt'))

    # save persistent z to cloud
    #if z_persistent is not None:
    #  persistent_z_gather = strategy.gather(z_persistent, axis=0)
    #  with open(os.path.join(exp_folder, 'checkpoints/persistent_z.ckpt'), 'wb') as f:
    #    pickle.dump(persistent_z_gather, f)
    #  upload_blob(config['gs_path'],
    #              os.path.join(exp_folder, 'checkpoints/persistent_z.ckpt'),
    #              os.path.join(exp_folder, 'checkpoints/persistent_z.ckpt'))
    #  # remove to save space
    #  os.remove(os.path.join(exp_folder, 'checkpoints/persistent_z.ckpt'))

    # save generator output
    if gen is not None:
      # gen model and optim
      gen.save_weights(os.path.join(gs_string, exp_folder, 'checkpoints/gen_{}.ckpt'.format(step+1)))
    # save gen optim
    #if gen_optim is not None:
    #  with open(os.path.join(exp_folder, 'checkpoints/gen_optim_{}.ckpt'.format(step+1)), 'wb') as f:
    #    pickle.dump(gen_optim.get_weights(), f)
    #  upload_blob(config['gs_path'],
    #              os.path.join(exp_folder, 'checkpoints/gen_optim_'+str(step+1)+'.ckpt'),
    #              os.path.join(exp_folder, 'checkpoints/gen_optim_'+str(step+1)+'.ckpt'))
