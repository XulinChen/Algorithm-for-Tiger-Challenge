import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch as T

class Main_Dataset(Dataset):
    def __init__(self, det_regions_lst, input_seg_net_regions_lst, detect_transform, image):
        self.idx_list = det_regions_lst
        self.region_list = input_seg_net_regions_lst
        self.my_transform = detect_transform
        self.wsi = image

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, ind):
        target_idx = self.idx_list[ind]
        x, y, tilesize = self.region_list[target_idx][:3]

        image_tile = self.wsi.getUCharPatch(
            startX=x, startY=y, width=tilesize, height=tilesize, level=0
        )
        imgs = Image.fromarray(image_tile)
        imgs = self.my_transform(imgs)

        info = T.LongTensor([target_idx]).squeeze()

        return imgs, info


def get_det_loaders(det_regions_lst, input_seg_net_regions_lst, detect_transform, image, batchsize=4):

    test_dataset = Main_Dataset(det_regions_lst, input_seg_net_regions_lst, detect_transform, image)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batchsize, shuffle=False, num_workers=0)

    print('Number of det samples: ', len(test_dataset))

    return test_loader





