from imagenet_dataset_light_lol import ImageFolderDataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import os
import torch

#!/usr/bin/env python

import getopt
import numpy
import PIL
import PIL.Image
import sys
import torch

assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 13) # requires at least pytorch version 1.3.0

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

def save_image_tensor(x, label, save_dir):
    # x is a tensor of shape BCHW
    # label is a tensor of shape B
    os.makedirs(save_dir, exist_ok=True)
    x = x.permute(0,2,3,1) # shape BHWC
    x = x.detach().cpu().numpy().astype(numpy.uint8)
    label = label.detach().cpu().numpy().astype(numpy.uint8)

    for i in range(label.shape[0]):
        # save_name = str(i).zfill(3) + '_label_' + str(label[i]).zfill(4) + '.png'
        save_name = str(i) + '_label_' + str('222') + '.jpg'
        Image.fromarray(x[i]).save(os.path.join(save_dir, save_name))

if __name__ == '__main__':
    import pdb
    path='/mnt/data/digital_content_aigc/feiben/DDPM_Beat_GAN/MD_image/teaser_more/AbdomenUS/processed_imgs'
    image_size1 = 512
    image_size2 = 512

    transform1 = transforms.Compose([transforms.Resize(image_size1), transforms.CenterCrop(image_size1)])
    dataset = ImageFolderDataset(path, permute=True, normalize=False, transform=transform1)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    transform2 = transforms.Compose([transforms.Resize(image_size2), transforms.CenterCrop(image_size2)])
    dataset2 = ImageFolderDataset(path, permute=True, normalize=False, transform=transform2)
    dataloader2 = torch.utils.data.DataLoader(dataset2, batch_size=1, shuffle=False, num_workers=1)

    imgs = []
    imgs_high = []
    labels = []
    for i, (data, data2) in enumerate(zip(dataloader, dataloader2)):
        x, y = data
        x2, y2 = data2

        imgs.append(x)
        imgs_high.append(x2)
        labels.append(y)

    imgs = torch.cat(imgs, dim=0)
    imgs_high = torch.cat(imgs_high, dim=0)
    labels = torch.cat(labels, dim=0)

    imgs = imgs.permute(0,2,3,1) # shape BHWC
    imgs = imgs.detach().cpu().numpy()

    imgs_high = imgs_high.permute(0,2,3,1) # shape BHWC
    imgs_high = imgs_high.detach().cpu().numpy().astype(numpy.uint8)

    labels = labels.detach().cpu().numpy().astype(numpy.uint8)

    file_name_high_1 = ('/mnt/data/digital_content_aigc/feiben/DDPM_Beat_GAN/MD_image/teaser_more/AbdomenUS_512_512_resolution_%d.npz' % image_size2)
    np.savez(file_name_high_1, imgs_high, labels)