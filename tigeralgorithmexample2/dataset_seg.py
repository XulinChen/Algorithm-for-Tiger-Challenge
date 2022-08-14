import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch as T

class Main_Dataset(Dataset):
    def __init__(self, input_seg_net_regions_lst, seg_transform, image):
        self.region_list = input_seg_net_regions_lst
        self.my_transform = seg_transform
        self.wsi = image

    def __len__(self):
        return len(self.region_list)

    def __getitem__(self, ind):
        x, y, tilesize = self.region_list[ind][:3]

        image_tile = self.wsi.getUCharPatch(
            startX=x, startY=y, width=tilesize, height=tilesize, level=0
        )
        imgs = Image.fromarray(image_tile)
        imgs = self.my_transform(imgs)

        info = T.LongTensor([ind]).squeeze()

        return imgs, info


def get_seg_loaders(input_seg_net_regions_lst, seg_transform, image, batchsize=3):

    test_dataset = Main_Dataset(input_seg_net_regions_lst, seg_transform, image)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batchsize, shuffle=False, num_workers=0)

    print('Number of seg samples: ', len(test_dataset))

    return test_loader





