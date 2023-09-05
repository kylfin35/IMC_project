import numpy as np
import pandas as pd
import glob
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.preprocessing import Normalizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
import os
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans
import random
from sklearn.neighbors import NearestNeighbors
import matplotlib
from scipy.ndimage import median_filter, gaussian_filter
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
import math
import copy
from sklearn import preprocessing
import pickle
import torchvision
from collections import Counter
import time, os, requests
from statistics import mode
import gc
import pytorch_lightning as pl
from pytorch_lightning.callbacks import StochasticWeightAveraging, GradientAccumulationScheduler
from scipy.stats import kruskal
from skimage.measure import block_reduce
from scipy.cluster.hierarchy import dendrogram, linkage
import sklearn
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import umap
from coclust.clustering import SphericalKmeans
from sklearn.preprocessing import StandardScaler
from bisect import bisect
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from skimage.io import imread
from collections import Counter
from lightly.models.modules import SwaVProjectionHead, SwaVPrototypes
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster
from lifelines.statistics import multivariate_logrank_test


class WholeImages(Dataset):
    def __init__(self, path_, keep_, norm=True):
        self.path_ = path_
        self.keep_ = keep_
        self.norm = norm
        self.img_ids_ = [a.strip('.tif') for a in os.listdir(self.path_)]
        self.img_ids = [a for a in self.img_ids_ if a in self.keep_]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        x = imread(os.path.join(self.path_, img_id) + '.tif')
        x = torch.tensor(x).float() / 255
        if self.norm:
            G_blur = torchvision.transforms.GaussianBlur(5, 1.5)
            x = G_blur(x)
            totals = torch.sum(x, dim=0).unsqueeze(0) + 1e-5
            totals = torch.repeat_interleave(totals, 18, dim=0)
            x = x / totals
            x = x / x.max()
        return x, img_id


class UniformTiling(Dataset):
    def __init__(self, imgs_path, tilesize, rm_blank, keep_, norm=True):
        self.tilesize = tilesize
        self.rm_blank_ = rm_blank
        self.imgs_path = imgs_path
        self.keep_ = keep_
        self.img_ids_ = [a.strip('.tif') for a in os.listdir(self.imgs_path)]
        self.img_ids = [a for a in self.img_ids_ if a in self.keep_]
        self.norm = norm

    def __len__(self):
        return len(self.img_ids)

    def tile_image_uniform(self, np_img, tilesize, rm_blank=True):
        if len(np_img.shape) == 3:
            np_img = np.expand_dims(np_img, 0)
        num_rows = np_img.shape[3] // tilesize
        num_cols = np_img.shape[2] // tilesize
        y_remainder = (np_img.shape[3] % tilesize) // 2
        x_remainder = (np_img.shape[2] % tilesize) // 2
        tmplst = []  # np.empty([num_rows*num_cols, np_img.shape[1], tilesize, tilesize])
        corners = []
        i = 0
        for row in range(num_rows):
            for col in range(num_cols):
                tile = np_img[:, :, x_remainder + col * tilesize:x_remainder + (col + 1) * tilesize,
                       y_remainder + row * tilesize:y_remainder + (row + 1) * tilesize]
                if rm_blank:
                    if tile.mean() < .01:
                        pass
                    else:
                        tmplst.append(tile)
                        corners.append([[col * tilesize + x_remainder, row * tilesize + y_remainder]])
                else:
                    tmplst.append(tile)
                    corners.append([[col * tilesize + x_remainder, row * tilesize + y_remainder]])
        return torch.tensor(np.concatenate(tmplst)), np.concatenate(corners)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        x = imread(os.path.join(self.imgs_path, img_id) + '.tif')
        x = torch.tensor(x).float() / 255
        if self.norm:
            G_blur = torchvision.transforms.GaussianBlur(5, 1.5)
            x = G_blur(x)
            totals = torch.sum(x, dim=0).unsqueeze(0) + 1e-5
            totals = torch.repeat_interleave(totals, 18, dim=0)
            x = x / totals
            x = x / x.max()
        tiles, corners = self.tile_image_uniform(x, self.tilesize, self.rm_blank_)
        return tiles.float(), corners, img_id


class GetEmbeddings:
    def __init__(self):
        pass

    def get_embeddings(self, loader, model, takelevel, ptdata_, tilesize, bit16_=False, intensity_control=False,
                       max_=10, count_cells=False):
        latentlst = []
        index_lst = []
        device = 'cuda'
        tilelst = []  ##
        cornerlst = []
        tiles_per_image = []
        names_lst = []
        tile_df = pd.DataFrame(columns=list(ptdata_.columns))
        cellcount_lst = []
        intensity_lst = []
        # names_lst_r = []
        for i in range(max_):
            if i % 10 == 0:
                print(i, '(model) images done')
            tiles, corners, name = loader[i]
            if count_cells:
                cellcount = self.get_cellcounts(corners, name, tilesize)
                cellcount_lst.append(cellcount)
            # name_ = [name]*len(tiles)
            # names_lst_r.append(name_)
            names_lst.append(name)
            cornerlst.append(corners)
            pt_row = ptdata_.loc[ptdata_['Key'] == name]
            repeated = pt_row.apply(lambda x: x.repeat(len(tiles)), axis=0)
            tile_df = pd.concat([tile_df, repeated], axis=0, ignore_index=True)

            intensity = torch.mean(tiles, dim=(2, 3)).detach().cpu().numpy()
            intensity_lst.append(intensity)
            for ii in range(len(tiles) // 128 + 1):
                if 128 * (ii + 1) < len(tiles):
                    tilebatch = tiles[ii * 128:(ii + 1) * 128]
                else:
                    tilebatch = tiles[ii * 128:]
                conv, z = model(tilebatch.to(device))
                if takelevel == 'conv':
                    latent = conv.flatten(1).detach().cpu().numpy()
                    if bit16_:
                        latent = latent.astype('float16')
                elif takelevel == 'out':
                    # latent = F.normalize(latent, dim=-1)
                    latent = z.detach().cpu().numpy()
                    if bit16_:
                        latent = latent.astype('float16')
                latentlst.append(latent)  # .flatten(1).detach().cpu().numpy())
            torch.cuda.empty_cache()
            gc.collect()
            tiles_per_image.append(len(tiles))
        if count_cells:
            uniques = self.get_uniques()
            cell_df = self.create_cellcount_df(uniques, cellcount_lst, len(tile_df))
            tile_df = pd.concat([tile_df, cell_df], axis=1)  # , ignore_index=True)
        embeddings = np.concatenate(latentlst)
        embedding_corners = np.concatenate(cornerlst)
        intensities = np.concatenate(intensity_lst)
        intensity_columns = ['ch' + str(i) for i in range(18)]
        tile_df.insert(1, 'X', embedding_corners[:, 0])  # , columns=['X','Y'])
        tile_df.insert(2, 'Y', embedding_corners[:, 1])
        for i in range(len(intensity_columns)):
            tile_df.insert(i + 3, intensity_columns[i], intensities[:, i])
        uniques[-1] = 'density'
        # tile_df.insert(0, 'Name', names_r)
        return embeddings, tiles_per_image, embedding_corners, names_lst, tile_df, uniques

    def get_cellcounts(self, corners_, pt, tilesize):

        cell_count_lst = []
        nms = loadmat(f'/content/drive/My Drive/LungData/LUAD_IMC_Segmentation/{pt}/nuclei_multiscale.mat')
        cell_map = nms['nucleiOccupancyIndexed']
        celltype = loadmat(f'/content/drive/My Drive/LungData/LUAD_IMC_CellType/{pt}.mat')
        celltypes = celltype['cellTypes']
        for ii in range(len(corners_)):
            corner = corners_[ii]
            window = cell_map[corner[0]:corner[0] + tilesize, corner[1]:corner[1] + tilesize]
            cells = np.unique(window)
            tmp = []
            for cell in cells:
                type_ = celltypes[cell - 1][0]
                if len(type_) == 0:
                    tmp.append('N/A')
                else:
                    tmp.append(type_[0])
            counts = Counter(tmp)
            cell_count_lst.append(counts)
        return cell_count_lst

    def get_uniques(self):
        celltype = loadmat(f'/content/drive/My Drive/LungData/LUAD_IMC_CellType/LUAD_V32B.mat')
        uniques_ = celltype['allLabels']
        uniques = ['N/A']
        for i in uniques_[0]:
            uniques.append(i[0])
        return uniques

    def create_cellcount_df(self, uniques, cellcounts_, tilelen):
        # reduced = trained_tsne.transform(patch_embeddings)
        # x,y = reduced[:,0], reduced[:,1]
        empty = np.zeros((tilelen, len(uniques)))
        j = 0
        for i in cellcounts_:
            for ii in i:
                for name, count in ii.items():
                    idx = uniques.index(name)
                    empty[j, idx] = int(count)
                j += 1
        df = pd.DataFrame(empty, columns=uniques)
        df['density'] = df.sum(axis=1)

        return df


class DimReduction:
    def __init__(self):
        pass

    def get_pca(self, dims, X):
        pca = PCA(dims)
        pca = pca.fit(X)
        latent_pca = pca.transform(X)
        pca1 = latent_pca[:, 0]
        pca2 = latent_pca[:, 1]
        return pca1, pca2, pca

    def get_umap(self, x):
        umap_ = umap.UMAP(random_state=np.random.randint(0, 100)).fit(x)
        umap_x, umap_y = umap_.embedding_[:, 0], umap_.embedding_[:, 1]
        return umap_x, umap_y, umap_

    def plot_reduced_dims(self, dim1, dim2, tiles_per_image_, n_images):
        fig = plt.figure(figsize=(10, 10))
        counter = 0
        idxs2 = [-1] * len(dim1)
        plt.rcParams["axes.grid"] = False
        for i in range(n_images):
            # l,r = counter*tiles_per_image_, (counter+1)*test_n
            l, r = counter, counter + tiles_per_image_[i]
            idxs2[l:r] = [counter] * (r - l)
            counter += tiles_per_image_[i]
        idxs = np.array(idxs2)
        x, y = dim1, dim2
        plt.scatter(x, y, c=idxs, cmap='tab20c', marker='.', s=2, alpha=.8)


class ClinicalData:
    def __init__(self):
        pass

    def kmeans_survival_data_temp(self, data, reduction_method, names, n_clusters, scale=False, max_=10,
                                  cluster_method_='kmeans'):
        x_, y_ = reduction_method + '_x', reduction_method + '_y'
        embeddings_ = np.concatenate([data[x_].values.reshape(-1, 1), data[y_].values.reshape(-1, 1)], axis=1)
        _, kmeans = self.create_embedding_clusters(n_clusters, embeddings_, cluster_method_)
        class_labels = self.assign_clusters(kmeans, embeddings_)
        # class_labels = SpectralClustering(n_clusters).fit_predict(embeddings_)
        data['kmeans_labels'] = class_labels
        E = np.zeros([len(names), n_clusters])
        # class_labels = class_labels.reshape(len(testloader), -1)
        for i in range(max_):
            name = names[i]
            pt = data.loc[data['Key'] == name].copy()  # name?
            class_counts = np.unique(pt['kmeans_labels'].copy().to_numpy(), return_counts=True)
            for ii in range(len(class_counts[0])):
                ii_ = class_counts[0][ii]
                E[i, ii_] = class_counts[1][ii] / (len(pt))  # -class_counts[1])
        ptDF = pd.DataFrame(columns=['Name', 'TIME', 'Survival', 'Age'] + [str(i) for i in range(n_clusters)])
        for i in range(max_):
            name = names[i]
            pt = data[data['Key'] == name][['TIME', 'Death', 'Age']].copy()
            pt2 = data[data['Key'] == name][
                ['Sex', 'Age', 'BMI', 'Smoking Status', 'Progression', 'Death', 'Pattern']].copy()
            time = pt['TIME'].values
            survival = pt['Death'].values
            age = pt['Age'].values
            sex = pt2['Sex'].values
            BMI = pt2['BMI'].values
            Smoking = pt2['Smoking Status'].values
            Prog = pt2['Progression'].values
            Pattern = pt2['Pattern'].values
            if np.isnan(survival):
                pass
            elif len(time) == 0:
                pass
            elif np.isnan(age):
                pass
            else:
                time, survival, age = time[0], survival[0], age[0]  # survival -1?
                sex, BMI, Smoking, Prog, Pattern = sex[0], BMI[0], Smoking[0], Prog[0], Pattern[0]
                classes = E[i, :].reshape(1, -1)
                classes = pd.DataFrame(classes, columns=list(ptDF)[4:])
                new_row = [name, time, survival, age] + [a for a in classes.values.astype(float).reshape(-1)]
                ptDF.loc[i + 1] = new_row
        #  if scale is True:
        #    scaler = StandardScaler()
        #    df[df.columns[2:]] = scaler.fit_transform(df[df.columns[2:]])
        #  df[df.columns[2:]] = df[df.columns[2:]]*scale_factor
        return ptDF, kmeans

    def create_embedding_clusters(self, num_clusters, X, cluster_method='kmeans'):
        if cluster_method == 'kmeans':
            cluster_ = KMeans(n_clusters=num_clusters, n_init=10, random_state=random.sample(range(100), 1)[0]).fit(X)
            centers = cluster_.cluster_centers_
        elif cluster_method == 'gaussian':
            cluster_ = GaussianMixture(n_components=num_clusters, random_state=random.sample(range(100), 1)[0]).fit(X)
            centers = 0
        return centers, cluster_

    def assign_clusters(self, kmeans_model, X):
        assignments = kmeans_model.predict(X)
        return assignments

    def univariate_coxph(self, df_, shuffle=False):
        for i in range(len(df_.columns) - 4):
            print(df_.columns[i + 4])
            df_2 = df_[['TIME', 'Survival', str(df_.columns[i + 4])]]
            cph = CoxPHFitter()
            cph.fit(df_2, duration_col='TIME', event_col='Survival',
                    show_progress=True)  # fit_options={'step_size':.2})
            cph.print_summary()
            if shuffle is True:
                shuffled_control = df_2.copy()
                for i in shuffled_control.columns:
                    if i == 'TIME':
                        print('pass')
                        pass
                    elif i == 'Survival':
                        pass
                    else:
                        shuffled_control[i] = np.random.permutation(df_2[i].values)
                cph2 = CoxPHFitter()
                cph2.fit(shuffled_control, duration_col='TIME', event_col='Survival',
                         show_progress=True)  # ,fit_options={'step_size':.2})
                cph2.print_summary()

    def patient_level_clustering(self, df_, n_clusters, scale=True):
        df = df_.copy()
        class_columns = [str(a) for a in range(n_clusters)]
        metadata = ['Name', 'TIME', 'Survival', 'Age']
        df = df.drop(metadata, axis=1)
        if scale:
            scaler = StandardScaler().fit(df[class_columns])
            df[class_columns] = scaler.transform(df[class_columns])
        link = sch.linkage(df, method='ward')
        dendrogram = sch.dendrogram(link)
        dend_x = dendrogram['ivl']
        dend_y = dendrogram['leaves_color_list']
        a = np.zeros((len(dend_x)))
        for i in range(len(a)):
            y = int(dend_y[i].strip('C'))
            a[int(dend_x[i])] = int(y - 1)
        df2 = df_.copy()
        df2['class_label'] = a
        result = multivariate_logrank_test(df2['TIME'], df2['class_label'], df2['Survival'])
        # result.test_statistic
        # result.p_value
        plt.figure()
        n_pt_classes = int(max(a)) + 1
        for i in range(n_pt_classes):
            kmf = KaplanMeierFitter()
            kmf.fit(df2['TIME'].loc[df2['class_label'] == i], df2['Survival'].loc[df2['class_label'] == i],
                    label=str(i))
            ax = kmf.plot(ci_show=False)
        ax.set_ylim([0.0, 1.0])
        print('number of patients in each class')
        for i in range(n_pt_classes):
            print(i, ':', len(df2.loc[df2['class_label'] == i]))

        result.print_summary()
        plt.figure()
        avg_class = []
        ##sns.clustermap(df, method='ward', robust=True)#, row_colors=list(df['class_label']))
        df3 = df.copy()
        df3['class_label'] = a
        for i in range(n_pt_classes):
            class_interest = df3.loc[df3['class_label'] == i].copy()
            class_interest = class_interest[class_columns]
            # plt.figure()
            fig, ax = plt.subplots()
            sns.heatmap(class_interest, ax=ax)
            ax.set_title(f"cluster ratio for patients of cluster {i}")
            plt.show()
            # plt.figure()
            tmp = class_interest.apply(lambda row: row.mean(), axis=0).values
            avg_class.append(tmp.reshape(1, -1))
        avg_class = np.concatenate(avg_class)
        fig, ax = plt.subplots()
        ax.set_title(f"mean cluster ratio by patient groups")
        sns.heatmap(avg_class, ax=ax)
        plt.show()

        return df2


class CellData:
    def __init__(self):
        pass

    def cell_cell_clustering(self, df, cell_types, n_clusters):
        cells_df = df[cell_types + ['kmeans_labels']].copy()
        new_df = pd.DataFrame(columns=cell_types)
        for i in range(n_clusters):
            k = cells_df.loc[cells_df['kmeans_labels'] == i]
            for cell in cell_types:
                if cell == 'kmeans_labels':
                    pass
                else:
                    avg = k[cell].mean()
                    new_df.loc[i, cell] = avg
        new_df.columns = new_df.columns.astype(str)
        scaler = StandardScaler()
        scaler.fit(new_df)
        new_df = pd.DataFrame(scaler.transform(new_df), columns=cell_types)
        fig, ax = plt.subplots()
        clustermap = sns.heatmap(new_df.astype(float), ax=ax)
        # clustermap.set_title('cell counts by latent cluster')
        ax.set_title('mean cell counts by latent cluster')
        plt.show()

    def cell_density_spread(self, tile_df, cell_types):
        lst = []
        x = tile_df['umap_x'].values
        y = tile_df['umap_y'].values
        mx = x - tile_df['umap_x'].values.mean()
        my = y - tile_df['umap_y'].values.mean()
        xy = np.concatenate((mx.reshape(-1, 1), my.reshape(-1, 1)), axis=1)
        total_spread = (np.linalg.norm(xy, axis=1) ** 2).mean()

        for celltype in cell_types[:-1]:
            print(celltype)
            counts = tile_df[celltype].values
            xdif = x - sum(x * counts) / sum(counts)
            ydif = y - sum(y * counts) / sum(counts)
            xy = np.concatenate((xdif.reshape(-1, 1), ydif.reshape(-1, 1)), axis=1)
            norm = np.linalg.norm(xy, axis=1) ** 2
            weighted_norm = np.sum(norm * counts) / np.sum(counts)
            out = weighted_norm / total_spread
            lst.append(out)
            print(out, '\n')
        print('average_spread =', sum(lst) / len(lst))


class ChooseClusters:
    def __init__(self):
        pass

    def get_elbow(self, latent):
        distortions = []
        inertias = []
        mapping1 = {}
        mapping2 = {}
        K = [2, 3, 7, 10, 25, 100]
        counter = 0
        for k in K:
            print('percent done: ', 100 * counter / len(K))
            counter += 1
            kmeans = KMeans(n_clusters=k).fit(latent)
            kmeans.fit(latent)
            distortions.append(sum(np.min(cdist(latent, kmeans.cluster_centers_,
                                                'euclidean'), axis=1)) / latent.shape[0])
            inertias.append(kmeans.inertia_)

            mapping1[k] = sum(np.min(cdist(latent, kmeans.cluster_centers_,
                                           'euclidean'), axis=1)) / latent.shape[0]
            mapping2[k] = kmeans.inertia_

        for key, val in mapping1.items():
            print(f'{key} : {val}')

        plt.plot(K, distortions, 'bx-')
        plt.xlabel('Values of K')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method using Distortion')
        plt.show()

        def get_silhouette(start, step, steps, alg, df, reduce_method):
            x = []
            sils = []
            latent_embs = np.concatenate(
                [df[reduce_method + '_x'].values.reshape(-1, 1), df[reduce_method + '_y'].values.reshape(-1, 1)],
                axis=1)
            for i in range(steps):
                n_ = step * i + start
                x.append(n_)
                print(100 * i / steps, '% done')
                if alg == 'kmeans':
                    cluster = KMeans(n_clusters=n_, random_state=np.random.RandomState()).fit(latent_embs)
                elif alg == 'gaussian mixture':
                    cluster = GaussianMixture(n_components=n_, random_state=np.random.RandomState()).fit(latent_embs)
                labels = cluster.predict(latent_embs)
                sil = silhouette_score(latent_embs, labels, metric='euclidean')
                sils.append(sil)
            plt.plot(x, sils)


class ImageReconstruction:
    def __init__(self):
        pass

    def kmeans_(self, n_clusters, umap_flag2):
        if umap_flag2 is True:
            centers, kmeans = create_embedding_clusters(n_clusters,umap_embeddings, False)
            class_labels = assign_clusters(kmeans, umap_embeddings)
        else:
            kmeans = SphericalKmeans(n_clusters=n_clusters, random_state=random.sample(range(100),1)[0])
            kmeans.fit(embeddings)
        return kmeans

    def pixel_tiling(self, i, size, step):
        img_idxs = [0]
        stepsize = []
        windows = []
        x_steps = (i.shape[1]-size)//step
        x_remainder = (i.shape[1]-size)%step//2
        y_steps = (i.shape[2]-size)//step
        y_remainder = (i.shape[2]-size)%step//2
        stepsize.append([x_steps,y_steps])
        counter = 0
        numerical = np.array(range(i.shape[1]*i.shape[2])).reshape(i.shape[1], i.shape[2])
        pixel_lst = list([[-1]]) * i.shape[1]*i.shape[2]
        for iy in range(y_steps):
            for ix in range(x_steps):
                x_left, x_right = x_remainder+(ix*step), x_remainder+(ix*step)+size
                y_top, y_bottom = y_remainder+(iy*step), y_remainder+(iy*step)+size
                windows.append(i[:, x_left:x_right, y_top:y_bottom])
                indeces = numerical[x_left:x_right, y_top:y_bottom]

                for id in indeces.flatten():
                    pixel_lst[id] = pixel_lst[id] + [counter]

                counter +=1
        windows = np.stack(windows)
        #if num_markers == 35:
        #windows = correct_channels(max_lst, rm_lst, windows)
        return torch.tensor(windows), pixel_lst

    def pixel_consensus(self, windows,elbow,pixel_lst_, img_id,kmeans_, image, level, umap_flag):
        batchsize_ = 300
        batches = len(windows)//batchsize_
        leftover = len(windows)%batchsize_
        tmp_lst = []
        for i in range(batches):
            window_ = windows[batchsize_*i:batchsize_*(i+1)]
            conv,out = clr_model(window_.to(device).float())
            if level == 'out':
                out = F.normalize(out.float(), dim=-1)
                ptxl = out.flatten(1).detach().cpu().numpy()
            elif level == 'conv':
                ptxl = conv.flatten(1).detach().cpu().numpy()
            if umap_flag == True:
                ptxl = trained_umap.transform(ptxl)
            tmp_lst.append(ptxl)
        ptxl = np.concatenate(tmp_lst)
        leftover_conv, leftover_out = clr_model(windows[-leftover:].to(device).float())
        if level == 'out':
            leftover = leftover_out.flatten(1).detach().cpu().numpy()
        elif level=='conv':
            leftover = leftover_conv.flatten(1).detach().cpu().numpy()
        if umap_flag == True:
            leftover = trained_umap.transform(leftover)
        ptxl = np.concatenate((ptxl, leftover))
        pxid = assign_clusters(kmeans_, ptxl)

        for i in range(len(pixel_lst_)):
            if pixel_lst_[i] == -1:
                pixel_lst_[i] = 0
            else:
                for ii in range(len(pixel_lst_[i])):
                    ix = pixel_lst_[i][ii]
                    id = pxid[ix]
                    pixel_lst_[i][ii] = id +1

                pixel_lst_[i] = Counter(pixel_lst_[i]).most_common()[0][0]

        classed_image = np.array(pixel_lst_).reshape(image.shape[1], image.shape[2])
        fig = plt.figure(figsize = (10,10))
        ax1 = fig.add_subplot(1,2,1)
        #ax1.legend(handles=range(elbow))
        classed_image[:elbow,0] = range(elbow)
        img1 =ax1.imshow(classed_image, cmap = 'tab20c')#(np.flipud(np.rot90(reshaped)), cmap=cmaps[cmap_id])
        ax2 = fig.add_subplot(1,2,2)
        ch1 = 8
        ch2 = 12
        ch3 = 16
        ch4 = 17
        a,b,c,d = image[ch1,:,:], image[ch2,:,:], image[ch3,:,:], image[ch4,:,:]
        a,b,c,d = a/a.max(), b/b.max(), c/c.max(), d/d.max()
        img2 = ax2.imshow(blend(a,b,c,d, cmap_blue, cmap_pink, cmap_green, cmap_orange))
        return classed_image

##reconstruction
# kmeans
# pixel tiling
# pixel reconstruction
# * show annotation
# ** (create function) comparing annotation to reconstruction


##exploratory
# show n closest tiles

##auxilary clinical analysis
# create complete dataframe
# do top cluster analysis
