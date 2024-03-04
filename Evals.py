import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
#from IMC_Dataset import ImageDataset
import torchvision
from sklearn.decomposition import PCA
import copy


def view_attn_heads(model_, img, pca_ = None, view=True, patch_size = 8):

  threshold = .1
  w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
  w_featmap = img.shape[-2] // patch_size
  h_featmap = img.shape[-1] // patch_size

  if len(img.shape) == 3:
    img = img.unsqueeze(0)
  elif len(img.shape) == 4:
    if img.shape[0] >1:
      img = img[0]
      img = img.unsqueeze(0)

  attentions = model_.get_last_selfattention(img.float())

  nh = attentions.shape[1] # number of head

  # we keep only the output patch attention
  attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

  if threshold is not None:
      # we keep only a certain percentage of the mass
      val, idx = torch.sort(attentions)
      val /= torch.sum(val, dim=1, keepdim=True)
      cumval = torch.cumsum(val, dim=1)
      th_attn = cumval > (1 - threshold)
      idx2 = torch.argsort(idx)
      for head in range(nh):
          th_attn[head] = th_attn[head][idx2[head]]
      th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
      # interpolate
      th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

  attentions = attentions.reshape(nh, w_featmap, h_featmap)
  attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().detach().numpy()

  # Visualize attentions
  if view is True:
    fig, axs = plt.subplots(nrows=nh, ncols=2, figsize=(16, 16))
    plt.axis('off')

    if pca_ is not None:
      view = pca_.transform(img[0].permute(1,2,0).reshape(-1, img.shape[1]))
      view = view.reshape(img.shape[-2], img.shape[-1], 3)

    else:
      if img.shape[1] > 3:
        view = img[0,:3,:,:]
        view = view.permute(1,2,0)
      else:
        view = img[0]
        view = view.permute(1,2,0)

    for i in range(nh):
        axs[i, 0].imshow(view)
        axs[i, 0].set_title(f'Image')
        axs[i, 1].imshow(attentions[i], cmap='gray')
        axs[i, 1].set_title(f'Attention map of head {i+1}')

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.show()

  return attentions


def stitch_attention_heads(model_, n, path_, tilesize2=128, data_norm=(None, None)):
    #input model, image number,
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model_.to(device)
    uniform_dset = ImageDataset(path_, n_tiles=0, delta=0, tilesize=tilesize2, resize_=224, strength='Weak', swav_=False,norm=False, n_neighbors=0, uniform_tiling=True, whole_image=False)
    tiles = uniform_dset[n][0]
    corners =  uniform_dset[n][1]
    x_steps = int(corners[:,0].max()//tilesize2) + 1
    y_steps = int(corners[:,1].max()//tilesize2) + 1
    x_width, y_width = int(x_steps*224), int(y_steps*224)
    stitched = np.zeros((6, x_width,  y_width))
    data_norm = torchvision.transforms.Normalize(mean = data_norm[0], std = data_norm[1])
    i = 0
    for y_ in range(y_steps):
      for x_ in range(x_steps):
        x,y = x_*224, y_*224
        input_ = tiles[i].unsqueeze(0)
        normed_input = data_norm(input_)
        normed_input = torch.clamp(normed_input, min=None, max=4)
        attention = view_attn_heads(model, normed_input.to(device), view=False)
        i += 1
        stitched[:, x:x+224, y:y+224] = attention

    #stitched = [cv2.resize(stitched[ii], (tilesize2, tilesize2)) for ii in range(6)]
    #stitched = np.stack(stitched)

    return stitched


def create_pca(full):

  pca = PCA(n_components=3)
  pca = pca.fit(np.transpose(full, (1,2,0)).reshape(-1, full.shape[0]))

  return pca


def transform_with_pca(img, pca=None):

  while len(img.shape) > 3:
    img = img[0]

  if pca is None:
    pca = create_pca(img)

  img = np.array(img)
  img2 = np.transpose(img, (1,2,0)).reshape(-1, img.shape[0])
  img2 = pca.transform(img2).reshape(img.shape[1], img.shape[2],3)
  return img2


def expand_input_dims(pretrained_model, input_dim, initialize_with= 'random'):

  multiplex_vit = copy.deepcopy(pretrained_model)

  # Create a new patch embedding layer
  original_patch_embed = multiplex_vit.patch_embed
  new_patch_embed = nn.Conv2d(input_dim, original_patch_embed.proj.out_channels,
                              kernel_size=original_patch_embed.proj.kernel_size,
                              stride=original_patch_embed.proj.stride,
                              padding=original_patch_embed.proj.padding,
                              bias=False)


  # Initialize new layer
  if initialize_with=='random':
    new_patch_embed.weight.data.normal_(0, 1)

  elif initialize_with=='avg':
    old_weights =  pretrained_model.patch_embed.proj.weight.data
    avg_weights = torch.mean(old_weights, dim=1, keepdim=True).repeat(1, input_dim, 1, 1)
    new_patch_embed.weight.data = avg_weights + torch.randn_like(avg_weights)/2

  elif initialize_with=='single':
    single_weights =  pretrained_model.patch_embed.proj.weight.data[:,2,:,:].unsqueeze(1)
    single_weights = single_weights.repeat(1, input_dim, 1, 1)
    new_patch_embed.weight.data = single_weights + torch.randn_like(single_weights)/2

  # Replace the original patch embedding layer with the new one
  multiplex_vit.patch_embed.proj = new_patch_embed
  return multiplex_vit