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


class augment_images(nn.Module):
    def __init__(self, tilesize):
        super().__init__()
        self.tilesize = tilesize
        blur = torchvision.transforms.GaussianBlur(3, sigma=(.3, .6))
        h_flip = torchvision.transforms.RandomHorizontalFlip(p=.5)
        v_flip = torchvision.transforms.RandomVerticalFlip(p=.5)
        # rotate = torchvision.transforms.RandomRotation(45)
        rand_apply = torchvision.transforms.RandomApply(torch.nn.ModuleList([blur]), p=.5)
        # rand_apply2 = torchvision.transforms.RandomApply(torch.nn.ModuleList([rotate]), p=.33)

        erase = torchvision.transforms.RandomErasing(p=.5, scale=(.1, .33), ratio=(.2, 5), value=0)
        ones = torchvision.transforms.RandomErasing(p=.5, scale=(.1, .33), ratio=(.2, 5), value=0)
        resizecrop_strong = torchvision.transforms.RandomResizedCrop((tilesize, tilesize), scale=(0.25, 4),
                                                                     ratio=(0.75, 1.3333))
        resizecrop_weak = torchvision.transforms.RandomResizedCrop((tilesize, tilesize), scale=(0.66, 1.33),
                                                                   ratio=(0.75, 1.3333))
        rand_apply_s_crop = torchvision.transforms.RandomApply(torch.nn.ModuleList([resizecrop_strong]), p=1)
        rand_apply_w_crop = torchvision.transforms.RandomApply(torch.nn.ModuleList([resizecrop_weak]), p=1)
        self.strong_transforms = torchvision.transforms.Compose([h_flip, v_flip, rand_apply, rand_apply_s_crop])
        self.weak_transforms = torchvision.transforms.Compose([h_flip, v_flip, rand_apply_w_crop, erase])

    def forward(self, frame, strength):
        if strength == 'Strong':
            frame = self.strong_transforms(frame)
            jitter = (torch.rand(frame.shape[1], 1, 1) * .6) + .7
            # jitter[jitter<.75] = 0
            if random.uniform(0, 1) < .66:
                thresh = np.random.normal(0, 1) * .1 + .4
                rand_idxs = np.random.choice(len(frame), size=5, replace=False)
                # frame[rand_idxs] = torch.where(frame[rand_idxs]>thresh, torch.ones_like(frame[rand_idxs]), torch.zeros_like(frame[rand_idxs]))
            frame = jitter.to(frame.device) * frame

        elif strength == 'Weak':
            frame = self.weak_transforms(frame)
            if random.uniform(0, 1) < .9:
                jitter = (torch.rand(frame.shape[1], 1, 1) * .2) + .9
                # jitter[jitter<.87] = 0
                frame = jitter.to(frame.device) * frame

        return frame


class ImageDataset(Dataset):
    def __init__(self, imgs_path, n_tiles, delta, tilesize, channel_weight=None, strength='Weak', swav_=False,
                 norm=True, n_neighbors=1):
        self.imgs_path = imgs_path
        if self.imgs_path[-3:] == 'npz':
            self.np_images, self.img_ids = self.process_np(self.imgs_path)
        else:
            self.img_ids = [a for a in os.listdir(imgs_path)]
        self.n_tiles = n_tiles
        self.delta = delta
        self.tilesize = tilesize
        self.strength = strength
        self.swav_ = swav_
        self.norm = norm
        self.n_neighbors = n_neighbors

    def proces_np(self, path_):
        data = np.load(path_, allow_pickle=True)
        np_images = data['X']
        img_ids = [a for a in range(len(data))]
        for i in range(len(images)):
            t = images[i] / np.max(images[i], axis=(1, 2)).reshape(len(images[i]), 1, 1)
            np_images[i] = t
        return np_images, img_ids

    def __len__(self):
        return len(self.img_ids)

    def tile_image(self, np_img, tilesize, n, delta, n_neighbors=1):
        if len(np_img.shape) == 3:
            np_img = np.expand_dims(np_img, 0)
        cornerlst = []
        neighbors_lst = []
        counter = 0
        anchors_ = np.empty([n, np_img.shape[1], tilesize, tilesize])
        neighbors_ = np.empty([n, np_img.shape[1], tilesize, tilesize])

        while counter < n:
            xl = random.sample(range(0, (np_img.shape[2] - tilesize - 1)), 1)[0]
            yl = random.sample(range(0, (np_img.shape[3] - tilesize - 1)), 1)[0]
            anchor = np_img[:, :, xl:xl + tilesize, yl:yl + tilesize]
            if anchor.mean() > .005:
                anchors_[counter] = anchor
                cornerlst.append([[xl, yl]])
                counter += 1
            else:
                pass

        for i in range(n_neighbors):
            neighbors_tmp = np.empty([n, np_img.shape[1], tilesize, tilesize])
            for e, corner in enumerate(cornerlst):
                xl, yl = corner[0]
                xdelta = int(np.rint(random.gauss(0, delta))) + xl
                ydelta = int(np.rint(random.gauss(0, delta))) + yl
                if xdelta < 0:
                    xdelta = 0
                elif xdelta + tilesize > np_img.shape[2]:
                    xdelta = np_img.shape[2] - tilesize - 1
                if ydelta < 0:
                    ydelta = 0
                elif ydelta + tilesize > np_img.shape[3]:
                    ydelta = np_img.shape[3] - tilesize - 1
                neighbor = np_img[:, :, xdelta:xdelta + tilesize, ydelta:ydelta + tilesize]
                neighbors_tmp[e] = neighbor
            neighbors_lst.append(neighbors_tmp)
        corners = np.concatenate(cornerlst)
        return anchors_, neighbors_lst, corners

    def __getitem__(self, idx):
        if self.imgs_path[-3:] == 'npz':
            x = self.np_images[idx]
        else:
            img_id = self.img_ids[idx]
            x = imread(os.path.join(self.imgs_path, img_id))
        x = torch.tensor(x) / x.max()
        if self.norm:
            G_blur = torchvision.transforms.GaussianBlur(5, 1.5)
            x = G_blur(x)
            totals = torch.sum(x, dim=0).unsqueeze(0) + 1e-5
            totals = torch.repeat_interleave(totals, x.shape[0], dim=0)
            x = x / totals
            x = x / x.max()

        if self.n_tiles == None:
            return x.squeeze()
        else:
            #    if self.strength == 'Strong':
            #        delta_ = self.delta
            #    elif self.strength == 'Weak':
            #        delta_ = self.delta
            anchors, neighbors, corners = self.tile_image(x, self.tilesize, self.n_tiles, self.delta, self.n_neighbors)
            anchors, neighbors = torch.tensor(np.array(anchors)).float(), torch.tensor(np.array(neighbors)).float()
            return anchors, neighbors, corners


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, res_channels, norm_type='Batchnorm'):
        super(Residual, self).__init__()
        if norm_type == 'Batchnorm':
            self.norm = nn.BatchNorm2d(res_channels)
        elif norm_type == 'Groupnorm':
            self.norm = nn.GroupNorm(res_channels, res_channels)
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=res_channels,
                      kernel_size=3, stride=1, padding=1, bias=False),
            self.norm, )
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


class save_module(nn.Module):
    def __init__(self, encoder_, mlp_):
        super(save_module, self).__init__()
        self.encoder_ = encoder_
        self.mlp_ = mlp_

    def forward(self, x):
        conv = self.encoder_(x)
        out = self.mlp_(conv)
        return conv, out


class lightning_clr(pl.LightningModule):
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
        if r2 < .33:
            xi = self.augmenter(xi, 'Weak')
        elif r2 > .66:
            xi = self.augmenter(xi, 'Strong')

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
