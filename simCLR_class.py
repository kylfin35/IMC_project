import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from skimage.io import imread
import random
import torchvision
import os
import gc
import pytorch_lightning as pl
from flash.core.optimizers import LARS
from torchvision import models

# This class provides augmentations for tiles
class Augmenter(nn.Module):
    def __init__(self, tile_size):
        super().__init__()
        self.tile_size = tile_size
        # augmentations below
        blur = torchvision.transforms.GaussianBlur(3, sigma=(.3, .6))
        h_flip = torchvision.transforms.RandomHorizontalFlip(p=.5)
        v_flip = torchvision.transforms.RandomVerticalFlip(p=.5)
        ## rotate = torchvision.transforms.RandomRotation(45)
        rand_apply = torchvision.transforms.RandomApply(torch.nn.ModuleList([blur]), p=.5)
        ## rand_apply2 = torchvision.transforms.RandomApply(torch.nn.ModuleList([rotate]), p=.33)
        ##erase = torchvision.transforms.RandomErasing(p=.5, scale=(.1, .33), ratio=(.2, 5), value=0)
        ##ones = torchvision.transforms.RandomErasing(p=.5, scale=(.1, .33), ratio=(.2, 5), value=0)
        resizecrop_strong = torchvision.transforms.RandomResizedCrop((tile_size, tile_size), scale=(0.25, 4),
                                                                     ratio=(0.75, 1.3333), antialias=True)
        resizecrop_weak = torchvision.transforms.RandomResizedCrop((tile_size, tile_size), scale=(0.66, 1.33),
                                                                   ratio=(0.75, 1.3333), antialias=True)
        rand_apply_s_crop = torchvision.transforms.RandomApply(torch.nn.ModuleList([resizecrop_strong]), p=1)
        rand_apply_w_crop = torchvision.transforms.RandomApply(torch.nn.ModuleList([resizecrop_weak]), p=1)
        # Apply transforms into strong and weak functions
        self.strong_transforms = torchvision.transforms.Compose([h_flip, v_flip, rand_apply, rand_apply_s_crop])
        self.weak_transforms = torchvision.transforms.Compose([h_flip, v_flip, rand_apply_w_crop])

    def forward(self, frame, strength):
        # Apply strong transforms
        if strength == 'Strong':
            frame = self.strong_transforms(frame)
            jitter = (torch.rand(frame.shape[1], 1, 1) * .6) + .7 # jitter between .7 - 1.3 x per channel
            # Allows changing pixels to 0 and 1 based on random threshold for randomly selected 5 channels
            if random.uniform(0, 1) < .75:  # apply jitter 75% of runs
                thresh = np.random.normal(0, 1) * .1 + .4
                rand_idxs = np.random.choice(len(frame), size=5, replace=False)
                ## frame[rand_idxs] = torch.where(frame[rand_idxs]>thresh, torch.ones_like(frame[rand_idxs]), torch.zeros_like(frame[rand_idxs]))
            frame = jitter.to(frame.device) * frame  # apply jitter
        # Apply weak transforms
        elif strength == 'Weak':
            frame = self.weak_transforms(frame)
            if random.uniform(0, 1) < .9:
                jitter = (torch.rand(frame.shape[1], 1, 1) * .4) + .8
                # jitter[jitter<.87] = 0
                frame = jitter.to(frame.device) * frame

        return frame


# This class processes tif and numpy hyperion image files and returns tiles
class ImageDataset(Dataset):
    def __init__(self, imgs_path, n_tiles, delta, tilesize, channel_weight=None, strength='Weak', swav_=False,
                 norm=True, n_neighbors=1):
        self.imgs_path = imgs_path
        if self.imgs_path[-3:] == 'npz':  # for numpy files
            self.np_images, self.img_ids = self.process_np(self.imgs_path)
        else:  # primarily tiff files
            self.img_ids = [a for a in os.listdir(imgs_path)]
        self.n_tiles = n_tiles
        self.delta = delta
        self.tilesize = tilesize
        self.strength = strength
        self.swav_ = swav_  # boolean
        self.norm = norm  # boolean
        self.n_neighbors = n_neighbors

    #for numpy images, pre-processes and returns as array
    def process_np(self, path_):
        data = np.load(path_, allow_pickle=True)
        np_images = data['X']
        img_ids = [a for a in range(len(np_images))]
        for i in range(len(np_images)):
            t = np_images[i] / np.max(np_images[i], axis=(1, 2)).reshape(len(np_images[i]), 1, 1)
            np_images[i] = t
        return np_images, img_ids

    def __len__(self):
        return len(self.img_ids)

    # This function takes an image in numpy_array format and returns (n tiles, n neighbors)
    # Neighbors are nearby tiles of same size chosen at random per delta variable
    def tile_image(self, np_img, tilesize, n, delta, n_neighbors=1):
        if len(np_img.shape) == 3:
            np_img = np.expand_dims(np_img, 0)
        cornerlst = []  # append top left corner of tiles
        neighbors_lst = []  #append neighbor tiles
        counter = 0  # counts valid (non-corner) tiles
        anchors_ = np.empty([n, np_img.shape[1], tilesize, tilesize])  # filled later
        neighbors_ = np.empty([n, np_img.shape[1], tilesize, tilesize])   # filled later

        # Only appends tiles and neighbors that are over certain intensity threshold
        # Will continue to randomly select locations until n valid tiles are chosen
        while counter < n:
            # Choose corner  along x axis and y axis
            # ensure that corner will not result in tile 'hanging off' edge
            xl = random.sample(range(0, (np_img.shape[2] - tilesize - 1)), 1)[0]
            yl = random.sample(range(0, (np_img.shape[3] - tilesize - 1)), 1)[0]
            anchor = np_img[:, :, xl:xl + tilesize, yl:yl + tilesize]  # anchor is tile selected from image
            if anchor.mean() > .005:  # only count tiles with valid signal intensity
                anchors_[counter] = anchor
                cornerlst.append([[xl, yl]])
                counter += 1
            else:
                pass

        # Collect neighbors
        # iterate through number neighbors. >1 neighbor can be chosen for space-smoothing effect
        for i in range(n_neighbors):
            neighbors_tmp = np.empty([n, np_img.shape[1], tilesize, tilesize])  # filled later
            for e, corner in enumerate(cornerlst):  # iterate through anchor locations
                xl, yl = corner[0]  # anchor corner
                # Choose nearby corner, normal distribution, scaled by delta
                xdelta = int(np.rint(random.gauss(0, delta))) + xl  # x axis new corner
                ydelta = int(np.rint(random.gauss(0, delta))) + yl  # y axis new corner
                # Ensure new corner does not 'hang off' edge
                if xdelta < 0:
                    xdelta = 0
                elif xdelta + tilesize > np_img.shape[2]:
                    xdelta = np_img.shape[2] - tilesize - 1
                if ydelta < 0:
                    ydelta = 0
                elif ydelta + tilesize > np_img.shape[3]:
                    ydelta = np_img.shape[3] - tilesize - 1
                neighbor = np_img[:, :, xdelta:xdelta + tilesize, ydelta:ydelta + tilesize] #from same image as tile
                neighbors_tmp[e] = neighbor
            neighbors_lst.append(neighbors_tmp)
        corners = np.concatenate(cornerlst)
        return anchors_, neighbors_lst, corners

    # This function determines number of channels in images, which varies by dataset
    def get_num_markers(self, dataset):
        a, _, _ = dataset[0]
        return a.shape[1]

    # This function normalizes image according to parameters given above
    def normalize_image(self, x):
        # Always apply gaussian blur
        G_blur = torchvision.transforms.GaussianBlur(5, 1.5)
        x = G_blur(x)
        for c in range(len(x)):
            x[c] = np.clip(np.array(x[c]), None, np.quantile(x[c], .99))
        nan = np.where(x == 0, np.nan, x)
        quant = np.nanquantile(nan, .999, axis=(1, 2))
        x = x / quant.reshape(-1, 1, 1)
        totals = torch.sum(x, dim=0).unsqueeze(0) + 1e-5
        totals = torch.repeat_interleave(totals, x.shape[0], dim=0)
        x = x / totals
        # x = x / quant.reshape(-1,1,1)
        x = x / x.max()
        return x

    def __getitem__(self, idx):
        if self.imgs_path[-3:] == 'npz':
            x = self.np_images[idx]
            x = np.clip(np.array(x), None, np.quantile(x, .99))
        else:
            img_id = self.img_ids[idx]
            if img_id[-3:] == 'iff':
                x = imread(os.path.join(self.imgs_path, img_id))
                x = np.clip(np.array(x), None, np.quantile(x, .99))
            elif img_id[-2:] == '.t':
                x = torch.load(os.path.join(self.imgs_path, img_id))
        x = torch.tensor(x) / x.max()
        if self.norm:
            x = self.normalize_image(x)

        anchors, neighbors, corners = self.tile_image(x, self.tilesize, self.n_tiles, self.delta, self.n_neighbors)
        anchors, neighbors = torch.tensor(np.array(anchors)).float(), torch.tensor(np.array(neighbors)).float()
        resize = torchvision.transforms.Resize((224, 224), antialias=True)
        neighbors = resize(
            neighbors.reshape(self.n_neighbors * self.n_tiles, anchors.shape[1], anchors.shape[2], anchors.shape[3]))
        anchors = resize(anchors)
        return anchors, neighbors.reshape(self.n_neighbors, self.n_tiles, anchors.shape[1], anchors.shape[2],
                                          anchors.shape[3]), corners


# This class provides residual blocks for custom neural network
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, res_channels, norm_type='Batchnorm'):
        super(Residual, self).__init__()
        # Choose batchnorm vs groupnorm. Groupnorm better for small batch size
        # Unclear which performs better at large batch sizes
        if norm_type == 'Batchnorm':
            self.norm = nn.BatchNorm2d(res_channels)
        elif norm_type == 'Groupnorm':
            self.norm = nn.GroupNorm(res_channels, res_channels)
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=res_channels,
                      kernel_size=3, stride=1, padding=1, bias=False),
            self.norm, ) # create basic block
        # nn.BatchNorm2d(res_channels),
        # nn.GroupNorm(4, res_channels),#, affine=False),
        # nn.ReLU(True),
        # nn.Conv2d(in_channels=res_channels,
        #           out_channels=out_channels,
        #           kernel_size=1, stride=1, bias=False),)

    def forward(self, x):
        return x  # + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, out_channels, res_layers, res_channels, norm_type):
        super(ResidualStack, self).__init__()
        self.res_layers = res_layers
        self._layers = nn.ModuleList([Residual(in_channels, out_channels, res_channels, norm_type)
                                      for _ in range(self.res_layers)])

    def forward(self, x):
        for i in range(self.res_layers):
            x = self._layers[i](x)
        x = F.relu(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, embedding_dim, norm_type):
        super(Encoder, self).__init__()
        self.affine_flag = False
        self.norm_type = norm_type
        self.initial_in = nn.Sequential(nn.Conv2d(in_channels, in_channels, 5, 1, 2), nn.ReLU())

        self._conv_1 = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                               out_channels=num_hiddens // 4,
                                               kernel_size=4,
                                               stride=2, padding=3, dilation=2), )
        self.norm1 = self.norm_function(norm_type, num_hiddens // 4)
        self.norm2 = self.norm_function(norm_type, num_hiddens // 4)
        self.norm3 = self.norm_function(norm_type, num_hiddens // 2)
        self.norm4 = self.norm_function(norm_type, num_hiddens // 2)
        self.norm5 = self.norm_function(norm_type, num_hiddens)
        self.norm6 = self.norm_function(norm_type, num_hiddens)
        self.norm7 = self.norm_function(norm_type, num_hiddens)
        self.bn4 = self.norm_function(norm_type, embedding_dim)

        self.res1 = ResidualStack(in_channels=num_hiddens // 4,
                                  out_channels=num_hiddens // 4,
                                  res_layers=num_residual_layers,
                                  res_channels=num_hiddens // 4,
                                  norm_type=self.norm_type)
        self._conv_2 = nn.Sequential(nn.Conv2d(in_channels=num_hiddens // 4,
                                               out_channels=num_hiddens // 2,
                                               kernel_size=4,
                                               stride=2, padding=3, dilation=2), )

        self.res2 = ResidualStack(in_channels=num_hiddens // 2,
                                  out_channels=num_hiddens // 2,
                                  res_layers=num_residual_layers,
                                  res_channels=num_hiddens // 2,
                                  norm_type=self.norm_type)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens // 2,
                                 out_channels=num_hiddens,
                                 kernel_size=2,
                                 stride=2, padding=1, dilation=2)

        self.res3 = ResidualStack(in_channels=num_hiddens,
                                  out_channels=num_hiddens,
                                  res_layers=num_residual_layers,
                                  res_channels=num_hiddens,
                                  norm_type=self.norm_type)
        self._conv_4 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=2,
                                 stride=2, padding=0)

        self._conv_5 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=embedding_dim,
                                 kernel_size=2,
                                 stride=2, padding=0)
        self.pre_vq = nn.Conv2d(num_hiddens, embedding_dim, 1, 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def norm_function(self, n_type, ch):
        if n_type == 'Groupnorm':
            return nn.GroupNorm(ch, ch)
        elif n_type == 'Batchnorm':
            return nn.BatchNorm2d(ch)

    def forward(self, inputs):
        x = inputs
        x = self.initial_in(inputs)
        x = self._conv_1(x)
        x = self.norm1(x)
        x = self.res1(x)
        x = self.norm2(x)
        x = self._conv_2(x)
        x = self.norm3(x)
        x = self.res2(x)
        x = self.norm4(x)
        x = self._conv_3(x)
        x = self.norm5(x)
        x = self.res3(x)
        x = self.norm6(x)
        x = self._conv_4(x)
        x = self.norm7(x)
        x = F.relu(x)
        x = self._conv_5(x)
        # x = self.pre_vq(x)
        # x = self.bn4(x)
        # x = self.tanh(x)
        return x


class MLP_head(nn.Module):
    def __init__(self, in_shape, latent_dims):
        super(MLP_head, self).__init__()
        self.linear = nn.Sequential(
            nn.BatchNorm1d(in_shape),
            nn.Linear(in_shape, latent_dims),
            nn.BatchNorm1d(latent_dims),
            nn.ReLU(),
            nn.Linear(latent_dims, latent_dims),
            #          nn.BatchNorm1d(latent_dims)
        )

    def forward(self, x):
        flat = x.flatten(1)
        out = self.linear(flat)
        return F.normalize(out, dim=-1)


class SaveModule(nn.Module):
    def __init__(self, encoder_, mlp_):
        super(SaveModule, self).__init__()
        self.encoder_ = encoder_
        self.mlp_ = mlp_

    def forward(self, x):
        conv = self.encoder_(x)
        out = self.mlp_(conv)
        return conv, out


class LightningCLR(pl.LightningModule):
    def __init__(self,
                 config,
                 encoder: nn.Module,
                 mlp: nn.Module,
                 loss_class: nn.Module,
                 augmenter: nn.Module,
                 batchsize=256,
                 augment_thresh=5,
                 ):
        super().__init__()
        self.encoder = encoder
        self.mlp = mlp
        self.loss_class = loss_class
        self.batchsize = batchsize
        self.lr = config['lr']
        self.momentum = config['momentum']
        self.loss_type = config['loss_fn']  # triplet or infoNCE
        self.config = config
        self.augment_thresh = augment_thresh
        self.augmenter = augmenter

    def forward(self, x):
        x = self.encoder(x)
        out = self.mlp(x)
        return x, out

    def configure_optimizers(self):
        if self.config['optim'] == 'LARS':
            self.lr = .3 * self.batchsize / 256
            optimizer = LARS(self.parameters(), self.lr, momentum=self.momentum)
        elif self.config['optim'] == 'RMSprop':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr, momentum=self.momentum)

        cos_anneal = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        decay = torch.optim.lr_scheduler.ExponentialLR(optimizer, .98)
        chained = torch.optim.lr_scheduler.ChainedScheduler([cos_anneal, decay])
        onecycle = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.lr, max_lr=10 * self.lr, step_size_up=100)
        lr_scheduler_config = {
            "scheduler": cos_anneal,
            "interval": "step",  # epoch or step
            "frequency": 1, }
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}

    def loss_fn(self, zi_, zj_, fn):
        if fn == 'triplet':
            triplet_loss = nn.TripletMarginLoss(swap=True)
            shuffle_idx = torch.randperm(zi_.size(0))
            zk = zi_[shuffle_idx, :]
            loss = triplet_loss(zi_, zj_, zk)
        elif fn == 'infoNCE':
            loss = self.loss_class(zi_, zj_)
        return loss

    def training_step(self, train_batch, batch_idx):
        gc.collect()
        epoch = self.trainer.current_epoch
        epochs = self.trainer.max_epochs
        xi, xj_, _ = train_batch
        if len(xi.shape) == 5:
            xi = torch.flatten(xi, 0, 1)
            # xj = torch.flatten(xj, 0, 1)

        r1 = random.uniform(0, 1)
        r2 = random.uniform(0, 1)
        ##  if r1 < .33:
        ##      xj = self.augmenter(xj, 'Weak')
        ##  elif r1 > .66:
        ##      xj = self.augmenter(xj, 'Strong')
        # if r1<.66:
        #   xi = self.augmenter(xi, 'Strong')
        # else:
        #if r2 < .51:
        #    xi = self.augmenter(xi, 'Weak')
        #elif r2 > .5:
        #    xi = self.augmenter(xi, 'Strong')

        conv = self.encoder(xi)
        zi = self.mlp(conv)
        tmp_output_lst = []
        for i in range(xj_.shape[1]):
            xj = xj_[:, i, :, :, :]
            if len(xj.shape) == 5:
                xj = torch.flatten(xj, 0, 1)
            xj = self.augmenter(xj, 'Strong')
            conv2 = self.encoder(xj)
            tmp_output_lst.append(conv2)
        all_outputs = torch.stack(tmp_output_lst)
        average_output = torch.mean(all_outputs, dim=0)
        zj = self.mlp(average_output)
        gc.collect()
        loss = self.loss_fn(zi, zj, self.loss_type)
        if batch_idx % 10 == 0:
            print(epoch, loss)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        xi, xj, _ = val_batch
        if len(xi.shape) == 5:
            xi = torch.flatten(xi, 0, 1)
            xj = torch.flatten(xj, 0, 1)
        conv = self.encoder(xi)
        zi = self.mlp(conv)
        conv2 = self.encoder(xj)
        zj = self.mlp(conv2)
        loss = self.loss_fn(zi, zj, self.loss_type)
        print('val loss', loss)
        return {"val_loss": loss}


class PretrainedResnet(nn.Module):
    def __init__(self, new_in_channels=10, latent_size=128):
        super(PretrainedResnet, self).__init__()
        self.new_in_channels, self.latent_size = new_in_channels, latent_size
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(self.new_in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.model = self.replace_batchnorm_with_groupnorm(self.model)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.latent_size)

       # self.model = models.densenet121(weights=None)
      #  self.layer = self.model.conv1


       # self.new_layer = nn.Conv2d(in_channels=self.new_in_channels,
        #                           out_channels=self.layer.out_channels,
         #                          kernel_size=self.layer.kernel_size,
          #                         stride=self.layer.stride,
           #                        padding=self.layer.padding,
            #                       bias=self.layer.bias)
       # self.expanded_model = self.expand_input_weights(self.model, self.new_layer)

    def replace_batchnorm_with_groupnorm(self, module):
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                num_groups = 32  # You can adjust the number of groups as needed
                new_layer = nn.GroupNorm(num_groups, child.num_features)
                setattr(module, name, new_layer)
            else:
                self.replace_batchnorm_with_groupnorm(child)
        return module
    def expand_input_weights(self, model_, new_layer_):
        iterator = 0
        self.new_layer.weight[:, :self.layer.in_channels, :, :].data = self.layer.weight.clone()
        new_layer_.weight[:, :self.layer.in_channels, :, :].data = self.layer.weight.clone()
        for i in range(self.new_in_channels - self.layer.in_channels):
            channel = self.layer.in_channels + i
            channel_iterator = iterator % 3
            new_layer_.weight[:, channel:channel + 1, :, :].data = self.layer.weight[:,
                                                                   channel_iterator:channel_iterator + 1, ::].clone()
            new_layer_.weight = nn.Parameter(new_layer_.weight)
            iterator += 1
        model_.conv1 = new_layer_
        return model_

    def forward(self, x):
        return self.model(x)
