import math
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

def get_all_regions_seg(dimensions, tile_size, foreground_thresh, tissue_mask, segmentation_writer):
    """Get the coordinates of regions which will be input to the seg net.

        NOTE
            If the number of foreground pixels in the region is less than the foreground threshold, the region will not be input to the seg net.
    """
    collected_regions_lst = []
    for y in tqdm(range(0, dimensions[1], tile_size)):
        for x in range(0, dimensions[0], tile_size):
            tissue_mask_tile = tissue_mask.getUCharPatch(
                startX=x, startY=y, width=tile_size, height=tile_size, level=0
            ).squeeze()
            tissue_mask_tile_img = Image.fromarray(tissue_mask_tile.astype(np.uint8))
            tissue_mask_256 = transforms.Resize(
                size=(tissue_mask_tile_img.size[1] // 8, tissue_mask_tile_img.size[0] // 8),
                interpolation=Image.NEAREST)(
                tissue_mask_tile_img)
            tissue_mask_256 = np.array(tissue_mask_256)
            foreground_num = np.sum(tissue_mask_256)

            '''
            # get the foreground number of tissue mask at level 3
            thumb_x1 = math.floor(x / 8)
            thumb_y1 = math.floor(y / 8)
            thumb_tile_size = tile_size // 8
            tissue_mask_256 = tissue_mask.getUCharPatch(
                startX=thumb_x1, startY=thumb_y1, width=thumb_tile_size, height=thumb_tile_size, level=3
            ).squeeze()
            foreground_num = np.sum(tissue_mask_256)
            '''
            # print('foreground num:', foreground_num)
            if foreground_num <= foreground_thresh:
                segmentation_mask_zero = np.zeros((tile_size, tile_size), dtype=int)
                segmentation_writer.write_segmentation(tile=segmentation_mask_zero, x=x, y=y)
            else:
                collected_regions_lst.append([x, y, tile_size, tile_size])

    return collected_regions_lst, segmentation_writer
