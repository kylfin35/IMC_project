import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.io import imread
import random
import torchvision
import os
import pickle
import pandas as pd
import cv2
from scipy.ndimage import gaussian_filter


# This class processes multiple IMC file types and returns pre-processed tiles of the image for a given path
class ImageDataset(Dataset):
    def __init__(self, imgs_path, n_tiles, delta, tilesize, resize_=None, strength='Weak', swav_=False,
                 norm=True, n_neighbors=1, uniform_tiling=False, whole_image=False):
        self.imgs_path = imgs_path  # path for files directory. Must contain ONLY image files
        if self.imgs_path[-3:] == 'npz':  # for handling numpy (lung)
            self.np_images, self.img_ids = self.process_np(self.imgs_path)  # process numpy
        else:
            self.img_ids = [a for a in os.listdir(imgs_path)]

        if self.imgs_path[-14:] == 'DLBCL_hyperion':  # For handling DLBCL images
            with open('/content/drive/MyDrive/DLBCL_Associated_data/keep_markers.pkl', 'rb') as f:
                self.keep_markers = pickle.load(f)
        self.n_tiles = n_tiles  # number of tiles per image
        self.delta = delta  # neighbor sample distance factor
        self.tilesize = tilesize  # h x w
        self.strength = strength  # augmentation strength
        self.swav_ = swav_  # if swav model
        self.norm = norm  # if normalizing according to Pixie paper
        self.n_neighbors = n_neighbors  # num neighbors (for latent smoothing)
        self.resize_ = resize_  # to resize tiles (pretrained resnet)
        self.uniform_tiling = uniform_tiling  # for testing. Evenly spaced tiles from image
        self.whole_image = whole_image  # for testing. Whole image, no tiling

    # this function handles lung Numpy file format
    def process_np(self, path_):
        data = np.load(path_, allow_pickle=True)  # load pickled file
        np_images = data['X']  # data is dictionary-like object. X is images
        img_ids = [a for a in range(len(np_images))]  # IDs are numerical, not names
        for i in range(len(np_images)):  # iterate thru images
            t = np_images[i] / np.max(np_images[i], axis=(1, 2)).reshape(len(np_images[i]), 1, 1)  # channel norm
            np_images[i] = t
        return np_images, img_ids

    # This function handles text csv format for DLBCL dataset
    def process_txt(self, path_, keep_):
        df = pd.read_csv(os.path.join(path_), sep='\t', usecols=['X', 'Y'] + keep_)  # read file, keep relevant channels
        df = df.apply(pd.to_numeric, errors='coerce')  # convert from str to float
        arr2d = pd.pivot_table(df, index='X', columns='Y', values=keep_).reindex(keep_, axis=1,
                                                                                 level=0).values  # convert from 2d to 3d by pixel value
        arr3d = np.reshape(arr2d, (arr2d.shape[0], arr2d.shape[1] // len(keep_), len(keep_)), order='F').transpose(2, 0,
                                                                                                                   1)  # reshape into proper 3d shape
        if np.isnan(arr3d).any():
            arr3d[np.isnan(arr3d)] = 0
        return arr3d

    def __len__(self):
        return len(self.img_ids)

    # This function tiles the image evenly for testing
    def tile_image_uniform(self, np_img, otsu, tilesize, rm_blank=False, thresh=.05):
        if len(np_img.shape) == 3:  # if [C, W, H]
            np_img = np.expand_dims(np_img, 0)  # [1, C, W, H]
        num_rows = np_img.shape[3] // tilesize  # num evenly spaced rows
        num_cols = np_img.shape[2] // tilesize  # num evenly spaced columns
        y_remainder = (np_img.shape[3] % tilesize) // 2  # space left on each side
        x_remainder = (np_img.shape[2] % tilesize) // 2
        tmplst = []
        corners = []
        nulls = []
        i = 0
        # Iterate thru rows and columns, filling with tiles
        for row in range(num_rows):
            for col in range(num_cols):
                tile = np_img[:, :, x_remainder + col * tilesize:x_remainder + (col + 1) * tilesize,
                       y_remainder + row * tilesize:y_remainder + (row + 1) * tilesize]

                if otsu is not None:  # Otsu determines if tile is empty or has cells
                    otsu_tile = otsu[x_remainder + col * tilesize:x_remainder + (col + 1) * tilesize,
                                y_remainder + row * tilesize:y_remainder + (row + 1) * tilesize]
                if rm_blank:  # removes empty tiles, improves clustering downstream
                    if otsu_tile.mean() < thresh:  # empty tiles
                        tile[:, :, :, :] = 0
                        nulls.append([1])  # labels this tile to be removed downstream
                    else:
                        nulls.append([0])  # labels this tile to not be removed from analysis
                    tmplst.append(tile)
                    corners.append([[col * tilesize + x_remainder, row * tilesize + y_remainder]])
                else:
                    tmplst.append(tile)
                    corners.append([[col * tilesize + x_remainder, row * tilesize + y_remainder]])
                    nulls.append([0])
        return torch.tensor(np.concatenate(tmplst)), np.concatenate(corners), np.concatenate(nulls)

    # This function rnadomly selects n tiles and corresponding neighbors from image
    def tile_image(self, np_img, tilesize, n, delta, n_neighbors=1, otsu_=None):
        if len(np_img.shape) == 3:
            np_img = np.expand_dims(np_img, 0)
        if otsu_ is not None:  # *currently only optimized for liver dataset. need to make channel a variable
            dna1 = cv2.normalize(np_img[0, 39, :, :], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                 dtype=cv2.CV_32F).astype(np.uint8)
            dna2 = cv2.normalize(np_img[0, 40, :, :], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                 dtype=cv2.CV_32F).astype(np.uint8)
            otsu = (cv2.threshold(dna1, 0, 1, cv2.THRESH_OTSU)[1] + cv2.threshold(dna2, 0, 1, cv2.THRESH_OTSU)[1]) / 2
        cornerlst = []
        neighbors_lst = []
        counter = 0
        anchors_ = np.empty([n, np_img.shape[1], tilesize, tilesize])
        neighbors_ = np.empty([n, np_img.shape[1], tilesize, tilesize])
        # Iterate thru num neighbors, only counting tiles that are non-empty
        while counter < n:
            xl = random.sample(range(0, (np_img.shape[2] - tilesize - 1)), 1)[0]
            yl = random.sample(range(0, (np_img.shape[3] - tilesize - 1)), 1)[0]
            anchor = np_img[:, :, xl:xl + tilesize, yl:yl + tilesize]
            if otsu_ is not None:
                otsu_tile = otsu[xl:xl + tilesize, yl:yl + tilesize]
                if otsu_tile.mean() > 0.05:
                    anchors_[counter] = anchor
                    cornerlst.append([[xl, yl]])
                    counter += 1
                else:
                    pass
            else:
                anchors_[counter] = anchor  # append anchor
                cornerlst.append([[xl, yl]])  # top left corner for future analysis
                counter += 1

                # Iterate thru number of neighbors per tile
        for i in range(n_neighbors):
            neighbors_tmp = np.empty([n, np_img.shape[1], tilesize, tilesize])
            for e, corner in enumerate(cornerlst):
                otsu_count = 0
                while otsu_count == 0:  # keeps track of empty tiles
                    xl, yl = corner[0]  # top left corner for anchor
                    xdelta = int(np.rint(random.gauss(0, delta))) + xl  # neighbor corner
                    ydelta = int(np.rint(random.gauss(0, delta))) + yl  # neighbor corner
                    if xdelta < 0:  # make sure there is no 'overhang'
                        xdelta = 0  # push to edege
                    elif xdelta + tilesize > np_img.shape[2]:  # make sure there is no 'overhang'
                        xdelta = np_img.shape[2] - tilesize - 1
                    if ydelta < 0:  # make sure there is no 'overhang'
                        ydelta = 0
                    elif ydelta + tilesize > np_img.shape[3]:  # make sure there is no 'overhang'
                        ydelta = np_img.shape[3] - tilesize - 1
                    neighbor = np_img[:, :, xdelta:xdelta + tilesize, ydelta:ydelta + tilesize]
                    if otsu_:
                        otsu_neighbor = otsu[xdelta:xdelta + tilesize, ydelta:ydelta + tilesize]
                        if otsu_neighbor.mean() > .05:  # only take non empty otsu
                            otsu_count += 1
                    else:
                        otsu_count += 1
                neighbors_tmp[e] = neighbor
            neighbors_lst.append(neighbors_tmp)
        corners = np.concatenate(cornerlst)
        return anchors_, neighbors_lst, corners

    # This function gets number of markers, which changes between datasets
    def get_num_markers(self, dataset):
        a, _, _ = dataset[0]
        return a.shape[1]

    # This function normalizes image, with or without pixel-norm
    def normalize_image(self, x, norm_scale=True):
        # first clip each channel independently
        for c in range(len(x)):
            x[c] = np.clip(np.array(x[c]), None, np.quantile(x[c], .99))
        x = gaussian_filter(x, (0, 1.5, 1.5)) # gaussian filter all images
        if norm_scale:  # pixel norm
            nan = np.where(x == 0, np.nan, x)  # create mask that doesn't acount for 0 values
            if bool(np.isnan(nan).all()): # for slice that is all 0
                pass
            else:
                quant = np.nanquantile(nan, .999, axis=(1, 2))  # along channel axis
                x = torch.from_numpy(x / quant.reshape(-1, 1, 1))  # divide out by 99.9% non-zero value for channel
                totals = torch.sum(x, dim=0).unsqueeze(0) + 1e-5  # reshape scale factor
                totals = torch.repeat_interleave(totals, x.shape[0], dim=0)  # sum of relative per channel intensity
                x = x / totals  # create pixel-level scaling
        return x / x.max()  # 0-1 norm

    def __getitem__(self, idx):

        #handles numpy images
        if self.imgs_path[-3:] == 'npz':
            x = self.np_images[idx]
            x = np.clip(np.array(x), None, np.quantile(x, .99))
        else:
            img_id = self.img_ids[idx]
            if img_id[-3:] in ['iff', 'tif']:
                x = imread(os.path.join(self.imgs_path, img_id))  # tiff files
                if np.isnan(x).any():
                    x[np.isnan(x)] = 0
            elif img_id[-3:] == 'txt':  # specifically for DLBCL
                x = self.process_txt(os.path.join(self.imgs_path, img_id), self.keep_markers)
        x = self.normalize_image(x, self.norm)  # all images processed
        if self.whole_image:  # for testing
            return x, img_id.strip('.tiff')
        else:
            if self.uniform_tiling: # for testing
                output, corners, nulls = self.tile_image_uniform(x, None, self.tilesize, rm_blank=False, thresh=0)
                return output, corners, img_id.strip('.tiff'), nulls  # evenly spaced tiles and location
            else:  # for training
                anchors, neighbors, corners = self.tile_image(x, self.tilesize, self.n_tiles, self.delta, self.n_neighbors)
                anchors, neighbors = torch.tensor(np.array(anchors)).float(), torch.tensor(np.array(neighbors)).float()
                if self.resize_ != None:  #resize function (resnet)
                    resize = torchvision.transforms.Resize((self.resize_, self.resize_), antialias=True)
                    neighbors = resize(
                        neighbors.reshape(self.n_neighbors * self.n_tiles, anchors.shape[1], anchors.shape[2],
                                        anchors.shape[3]))
                    anchors = resize(anchors)
                return anchors, neighbors.reshape(self.n_neighbors, self.n_tiles, anchors.shape[1], anchors.shape[2],
                                                anchors.shape[3]), corners
