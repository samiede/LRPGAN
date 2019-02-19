import os
import sys
import shutil
import numpy as np
import errno
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from IPython import display
import torch
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

if torch.cuda.is_available():
    plt.switch_backend('agg')
import PIL, PIL.Image

'''
    TensorBoard Data will be stored in './runs' path
'''

lowest = -1.0
highest = 1.0


class Logger:
    epoch = 0
    batch = 0

    def __init__(self, model_name, data_name, dir_name):
        self.model_name = model_name
        self.data_name = data_name

        self.comment = '{}_{}'.format(model_name, data_name)
        self.data_subdir = '{}/{}'.format(dir_name, data_name)
        self.log_subdir = '{}/{}'.format(dir_name, 'runs/')

        out_dir = '{}'.format(self.data_subdir)
        Logger._make_fresh_dir(out_dir)

        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.log_subdir, comment=self.comment)

    def log(self, d_error, g_error, epoch, n_batch, num_batches):

        # var_class = torch.Tensor
        # if type(d_error.data) == var_class:
        #     d_error = d_error.data.cpu().numpy()
        # if type(g_error.data) == var_class:
        #     g_error = g_error.data.cpu().numpy()

        step = Logger._step(epoch, n_batch, num_batches)
        self.writer.add_scalar(
            '{}/D_error'.format(self.comment), d_error, step)
        self.writer.add_scalar(
            '{}/G_error'.format(self.comment), g_error, step)

    def log_images(self, images, relevance, num_images, epoch, n_batch, num_batches,
                   printdata, format='NCHW', normalize=True, noLabel=False):
        """
        input images are expected in format (NCHW)
        """

        relevance = visualize(relevance.cpu().numpy() if torch.cuda.is_available()
                              else relevance.numpy(), heatmap)

        if type(images) == np.ndarray:
            images = torch.from_numpy(images)
        if type(relevance) == np.ndarray:
            relevance = torch.from_numpy(relevance)

        if format == 'NCHW':
            relevance = relevance.permute(0, 3, 1, 2)
        step = Logger._step(epoch, n_batch, num_batches)
        img_name = '{}/images{}'.format(self.comment, '')

        # concat images
        if images.size(1) == 1:
            images = images.repeat(1, 3, 1, 1)
        if torch.cuda.is_available():
            images = images.cuda()
            relevance = relevance.cuda()

        # concat images and relevance in comb pattern
        images_comb = torch.Tensor()
        for pair in zip(images, relevance):
            comb = torch.cat((pair[0].unsqueeze(0), pair[1].unsqueeze(0)))
            images_comb = torch.cat((images_comb, comb))
        images = images_comb

        # Make horizontal grid from image tensor
        horizontal_grid = vutils.make_grid(
            images, normalize=normalize, scale_each=True, pad_value=1)
        # Make vertical grid from image tensor
        nrows = int(np.sqrt(num_images))
        grid = vutils.make_grid(
            images, nrow=2, normalize=True, scale_each=True, pad_value=1)

        # Add horizontal images to tensorboard
        self.writer.add_image(img_name, horizontal_grid, step)

        # Save plots
        self.save_torch_images(horizontal_grid, grid, epoch, n_batch, images_comb, printdata, noLabel)

    def save_torch_images(self, horizontal_grid, grid, epoch, n_batch, images, printdata, noLabel,
                          plot_horizontal=False):
        out_dir = '{}'.format(self.data_subdir)
        Logger._make_dir(out_dir)
        comment = '{:.4f}-{:.4f}'.format(printdata['test_prob'], printdata['real_test_prob'])

        if noLabel:
            # Plot and save horizontal
            if plot_horizontal:
                fig = plt.figure(figsize=(32, 16), facecolor='white')
                if torch.cuda.is_available():
                    horizontal_grid = horizontal_grid.cpu()
                plt.imshow(np.moveaxis(horizontal_grid.numpy(), 0, -1))
                plt.axis('off')
                display.display(plt.gcf())
                self._save_images(fig, epoch, n_batch, 'hori_')
                plt.close()

            # Save squared

            fig = plt.figure(figsize=(32, 32), facecolor='white')
            if torch.cuda.is_available():
                grid = grid.cpu()
            plt.imshow(np.moveaxis(grid.numpy(), 0, -1))
            plt.axis('off')
            self._save_images(fig, epoch, n_batch, comment=comment)
            plt.close()

        else:
            self._save_subplots(images, printdata, epoch, n_batch, comment=comment)

    def _save_subplots(self, images, printdata, epoch, n_batch, comment=''):
        out_dir = '{}'.format(self.data_subdir)
        Logger._make_dir(out_dir)
        if torch.cuda.is_available():
            images = images.cpu()

        num_plots = images.size(0) // 2
        cols = 2
        fig, axarr = plt.subplots(num_plots, cols)
        fig = plt.gcf()
        fig.set_size_inches(64, 64)
        index = 0
        for n in range(0, num_plots):
            image = vutils.make_grid(images[index], normalize=True, scale_each=True, pad_value=1)
            axarr[n, 0].imshow(np.moveaxis(image.numpy(), 0, -1))
            axarr[n, 0].axis('off')
            if n % 2 == 0:
                axarr[n, 0].set_title('{:.6f} / {:.6f}'.format(printdata['test_prob'], printdata['test_result']),
                                      fontsize=50)
            else:
                axarr[n, 0].set_title(
                    '{:.6f} / {:.6f}'.format(printdata['real_test_prob'], printdata['real_test_result']), fontsize=50)
            ttl = axarr[n, 0].title
            ttl.set_position([.5, 1.05])

            image = vutils.make_grid(images[index + 1], scale_each=True, pad_value=1)
            data = np.moveaxis(image.numpy(), 0, -1)
            axarr[n, 1].imshow(data)
            ttl = axarr[n, 1].title
            ttl.set_position([.5, 1.05])

            if n % 2 == 0:
                axarr[n, 1].set_title('{:.5f} / {:.5f}'.format(printdata['min_test_rel'], printdata['max_test_rel']),
                                      fontsize=50)
            else:
                axarr[n, 1].set_title('{:.5f} / {:.5f}'.format(printdata['min_real_rel'], printdata['max_real_rel']),
                                      fontsize=50)
            axarr[n, 1].axis('off')

            index += 2

        fig.savefig('{}/epoch_{}_batch_{}_{}.png'.format(out_dir, epoch, n_batch, comment), dpi=25)
        fig.savefig('{}/epoch_{}_batch_{}_{}.pdf'.format(out_dir, epoch, n_batch, comment), dpi=100)
        plt.close()

    def _save_images(self, fig, epoch, n_batch, comment=''):
        out_dir = '{}'.format(self.data_subdir)
        # Logger._make_dir(out_dir)
        fig.savefig('{}/epoch_{}_batch_{}_{}.png'.format(out_dir, epoch, n_batch, comment), dpi=50)
        fig.savefig('{}/epoch_{}_batch_{}_{}.pdf'.format(out_dir, epoch, n_batch, comment), dpi=100)

    @staticmethod
    def display_status(epoch, num_epochs, n_batch, num_batches, d_error, g_error, d_pred_real, d_pred_fake):

        # var_class = torch.Tensor
        # if type(d_error.data) == var_class:
        #     d_error = d_error.detach().cpu().item()
        # if type(g_error.data) == var_class:
        #     g_error = g_error.detach().cpu().item()
        # if type(d_pred_real.data) == var_class:
        #     d_pred_real = d_pred_real.data
        # if type(d_pred_fake.data) == var_class:
        #     d_pred_fake = d_pred_fake.data

        print('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(
            epoch, num_epochs, n_batch, num_batches)
        )
        print('Discriminator Loss: {:.4f}, Generator Loss: {:.4f}'.format(d_error, g_error))
        print('D(x): {:.4f}, D(G(z)): {:.4f} \n'.format(d_pred_real.mean(), d_pred_fake.mean()))

    def save_models(self, generator, discriminator, epoch):
        out_dir = './data/models/{}'.format(self.data_subdir)
        Logger._make_dir(out_dir)
        torch.save(generator.state_dict(),
                   '{}/G_epoch_{}'.format(out_dir, epoch))
        torch.save(discriminator.state_dict(),
                   '{}/D_epoch_{}'.format(out_dir, epoch))

    def close(self):
        self.writer.close()

    # Private Functionality

    @staticmethod
    def _step(epoch, n_batch, num_batches):
        return epoch * num_batches + n_batch

    @staticmethod
    def _make_dir(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    @staticmethod
    def _make_fresh_dir(directory):
        try:
            shutil.rmtree(directory)
        except OSError:
            pass
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    def save_image_batch(self, images, num):

        out_dir_pdf = '{}/{}/pdf'.format(self.data_subdir, 'generated')
        Logger._make_dir(out_dir_pdf)
        out_dir_png = '{}/{}/png'.format(self.data_subdir, 'generated')
        Logger._make_dir(out_dir_png)

        for n in range(0, images.size(0)):
            fig = plt.gcf()
            fig.set_size_inches(32, 32)
            image = vutils.make_grid(images[n], normalize=True, scale_each=True, pad_value=1)
            plt.imshow(np.moveaxis(image.detach().numpy(), 0, -1))
            plt.axis('off')

            if num:
                name = num
            else:
                name = n

            fig.savefig('{}/{}.png'.format(out_dir_png, name), dpi=100)
            fig.savefig('{}/{}.pdf'.format(out_dir_pdf, name), dpi=100)
            plt.close()

        vutils.save_image(images, '{}/{}.png'.format(out_dir_png, 'all_samples'), nrow=int(np.sqrt(images.size(0))), pad_value=1, normalize=True, scale_each=True)

    def save_heatmap_batch(self, images, relevance, probability, relu_result, num):

        min = []
        max = []
        for i in range(len(relevance)):
            min.append(torch.min(relevance[i]))
            max.append(torch.max(relevance[i]))

        relevance = visualize(relevance.cpu().numpy() if torch.cuda.is_available()
                              else relevance.numpy(), heatmap)
        if type(relevance) == np.ndarray:
            relevance = torch.from_numpy(relevance)
        if torch.cuda.is_available():
            images = images.cpu().detach()
            relevance = relevance.cpu().detach()

        # concat images and relevance in comb pattern
        relevance = relevance.permute(0, 3, 1, 2)
        images_comb = torch.Tensor()
        for pair in zip(images, relevance):
            comb = torch.cat((pair[0].unsqueeze(0), pair[1].unsqueeze(0)))
            images_comb = torch.cat((images_comb, comb))
        images = images_comb

        out_dir_pdf = '{}/{}/pdf'.format(self.data_subdir, 'discriminated')
        Logger._make_dir(out_dir_pdf)
        out_dir_png = '{}/{}/png'.format(self.data_subdir, 'discriminated')
        Logger._make_dir(out_dir_png)


        num_plots = images.size(0) // 2
        cols = 2
        fig, axarr = plt.subplots(num_plots, cols)
        fig = plt.gcf()
        fig.set_size_inches(32, num_plots * 32)
        index = 0
        for n in range(0, num_plots):
            image = vutils.make_grid(images[index], normalize=True, scale_each=True, pad_value=0)
            image = np.moveaxis(image.detach().numpy(), 0, -1)
            if num_plots > 1:
                ax0 = axarr[n, 0]
                ax1 = axarr[n, 1]
            else:
                ax0 = axarr[n]
                ax1 = axarr[n + 1]

            ax0.imshow(image)
            ax0.axis('off')
            ax0.set_title('{:.6f} / {:.6f}'.format(probability[n], relu_result[n]),
                          fontsize=50)

            ttl = ax0.title
            ttl.set_position([.5, 1.05])

            image = vutils.make_grid(images[index + 1], scale_each=True, pad_value=0)
            data = np.moveaxis(image.detach().numpy(), 0, -1)
            ax1.imshow(data)
            ttl = ax1.title
            ttl.set_position([.5, 1.05])

            ax1.set_title('{:.5f} / {:.5f}'.format(min[n], max[n]),
                          fontsize=50)
            ax1.axis('off')

            index += 2
        # else:
        #     fig = plt.figure(figsize=(32, 16), facecolor='white')
        #     image = vutils.make_grid(images, normalize=True, scale_each=True, pad_value=1)
        #     plt.imshow(np.moveaxis(image.detach().numpy(), 0, -1))
        #     plt.axis('off')
        #     plt.gcf()

        fig.savefig('{}/{}.png'.format(out_dir_png, num), dpi=50)
        fig.savefig('{}/{}.pdf'.format(out_dir_pdf, num), dpi=100)
        plt.close()

    @staticmethod
    def save_intermediate_heatmap(relevance, name):
        i = 0
        directory_name = os.path.basename(sys.argv[0]).split('.')[0]
        out_dir = './output/{}/intermediate/epoch {}/batch {}'.format(directory_name, str(Logger.epoch), str(Logger.batch))

        Logger._make_dir(out_dir)

        relevance = visualize(relevance.cpu().numpy() if torch.cuda.is_available()
                              else relevance.numpy(), heatmap)
        if type(relevance) == np.ndarray:
            relevance = torch.from_numpy(relevance)

        relevance = relevance.squeeze(0)
        fig = plt.figure(figsize=(32, 32), facecolor='white')
        if torch.cuda.is_available():
            relevance = relevance.cpu()
        plt.imshow(relevance.numpy())
        plt.axis('off')

        while os.path.exists('{}/_layer_{}_({}).png'.format(out_dir, name, Logger.getMarker(i))):
            i += 1

        fig.savefig('{}/_layer_{}_({}).png'.format(out_dir, name, Logger.getMarker(i)), dpi=50)
        plt.close()

    @staticmethod
    def getMarker(i):
        if i == 0:
            return 'g'
        else:
            return 'r'


# --------------------------------------
# Color maps ([-1,1] -> [0,1]^3)
# --------------------------------------

def heatmap(x):
    x = x[..., np.newaxis]

    # positive relevance
    hrp = 0.9 - np.clip(x - 0.3, 0, 0.7) / 0.7 * 0.5
    hgp = 0.9 - np.clip(x - 0.0, 0, 0.3) / 0.3 * 0.5 - np.clip(x - 0.3, 0, 0.7) / 0.7 * 0.4
    hbp = 0.9 - np.clip(x - 0.0, 0, 0.3) / 0.3 * 0.5 - np.clip(x - 0.3, 0, 0.7) / 0.7 * 0.4

    # negative relevance
    hrn = 0.9 - np.clip(-x - 0.0, 0, 0.3) / 0.3 * 0.5 - np.clip(-x - 0.3, 0, 0.7) / 0.7 * 0.4
    hgn = 0.9 - np.clip(-x - 0.0, 0, 0.3) / 0.3 * 0.5 - np.clip(-x - 0.3, 0, 0.7) / 0.7 * 0.4
    hbn = 0.9 - np.clip(-x - 0.3, 0, 0.7) / 0.7 * 0.5

    r = hrp * (x >= 0) + hrn * (x < 0)
    g = hgp * (x >= 0) + hgn * (x < 0)
    b = hbp * (x >= 0) + hbn * (x < 0)

    return np.concatenate([r, g, b], axis=-1)


def graymap(x):
    x = x[..., np.newaxis]
    return np.concatenate([x, x, x], axis=-1) * 0.5 + 0.5


# --------------------------------------
# Visualizing data
# --------------------------------------

def visualize(x, colormap):
    N = len(x)
    assert (N <= 16)
    try:
        x = colormap(x / np.abs(x).max())
    except ZeroDivisionError:
        x = colormap(x)

    # Create a mosaic and upsample
    if len(x.shape) <= 3:
        x = x.reshape([N, int(np.sqrt(x.shape[1])), int(np.sqrt(x.shape[1])), 3])
    else:
        x = x.reshape([N, x.shape[2], x.shape[3], 3])
    return x

    # x = np.pad(x, ((0, 0), (0, 0), (2, 2), (2, 2), (0, 0)), 'constant', constant_values=1)
    # x = x.transpose([0, 2, 1, 3, 4]).reshape([1 * 32, N * 32, 3])
    # x = np.kron(x, np.ones([2, 2, 1]))
    # PIL.Image.fromarray((x * 255).astype('byte'), 'RGB').save('./data/images/VGAN/MNIST/' + name)


# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)
    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))
