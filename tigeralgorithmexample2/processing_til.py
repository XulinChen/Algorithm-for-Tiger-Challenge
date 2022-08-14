from pathlib import Path
from typing import List
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
from tqdm import tqdm
from PIL import Image
import time
from .seg_config import seg_config
from .seg_config import update_seg_config
from .seg_hrnet import get_seg_model

from .det_config import det_config
from .det_config import update_det_config
from .det_hrnet import get_cell_det_net
# from .anchors import generate_default_anchor_maps, hard_nms
from .get_input_network_regions import get_all_regions_seg
from .dataset_seg import get_seg_loaders
from .dataset_det import get_det_loaders
import torch
from torch.nn import DataParallel
from torchvision import transforms
import os
from .gcio import (
    TMP_DETECTION_OUTPUT_PATH,
    TMP_SEGMENTATION_OUTPUT_PATH,
    TMP_TILS_SCORE_PATH,
    copy_data_to_output_folders,
    get_image_path_from_input_folder,
    get_tissue_mask_path_from_input_folder,
    initialize_output_folders,
)

from .rw import (
    READING_LEVEL,
    WRITING_TILE_SIZE,
    DetectionWriter,
    SegmentationWriter,
    TilsScoreWriter,
    open_multiresolutionimage_image,
)

def get_thresh_params(all_foreground_num):
    if all_foreground_num <= 1000000:
        seg_scale = 0.75
        foreground_thresh = 7000
        tumor_num_thresh = 3000
    elif (all_foreground_num > 1000000) and (all_foreground_num <= 1800000):
        seg_scale = 0.7
        foreground_thresh = 9000
        tumor_num_thresh = 3500
    elif (all_foreground_num > 1800000) and (all_foreground_num <= 2500000):
        seg_scale = 0.65
        foreground_thresh = 10000
        tumor_num_thresh = 4000
    else:
        seg_scale = 0.6
        foreground_thresh = 10000
        tumor_num_thresh = 4500
    return seg_scale, foreground_thresh, tumor_num_thresh

def get_seg_num(seg_img_array_256):
    seg_pred_256 = np.copy(seg_img_array_256)

    tumor_num = len(np.where(seg_pred_256 == 1)[0])
    inflame_s_num = len(np.where(seg_img_array_256 == 6)[0])
    all_num = seg_img_array_256.shape[0] * seg_img_array_256.shape[1]
    DCI_num = len(np.where(seg_img_array_256 == 3)[0])
    healthy_grand_num = len(np.where(seg_img_array_256 == 4)[0])

    return DCI_num, healthy_grand_num, tumor_num, inflame_s_num, all_num


def process_til():
    print(f"Pytorch GPU available: {torch.cuda.is_available()}")
    """get the seg model"""
    config_path, checkpoint_path = Path('/home/usr/models/seg_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'), Path(
        '/home/usr/models/tiger_seg_model_new_089_202204027.ckpt')
    seg_ckpt = torch.load(checkpoint_path)  # torch.load(checkpoint_path, map_location=torch.device('cpu'))
    update_seg_config(seg_config, config_path)

    # ====loading model=====
    print('Loading seg model...')
    seg_net = get_seg_model(seg_config)
    seg_net.load_state_dict(seg_ckpt['net_state_dict'])
    seg_net = seg_net.cuda()
    seg_net = DataParallel(seg_net)
    seg_net.eval()

    '''get the detect model'''
    detect_config_path, detect_checkpoint_path = Path('/home/usr/models/lymcells_detect_tiger_hrnet_w18.yaml'), Path(
        '/home/usr/models/lymcells_detect_256_384_768_768_079.ckpt')
    detect_ckpt = torch.load(
        detect_checkpoint_path)  # torch.load(detect_checkpoint_path, map_location=torch.device('cpu'))
    update_det_config(det_config, detect_config_path)
    print('Loading detection model...')
    det_net = get_cell_det_net(det_config)
    det_net.load_state_dict(detect_ckpt['net_state_dict'])
    det_net = det_net.cuda()
    det_net = DataParallel(det_net)
    det_net.eval()

    """Processes a test slide"""
    level = READING_LEVEL  # 0
    tile_size = WRITING_TILE_SIZE  # should be a power of 2

    initialize_output_folders()

    # get input paths
    image_path = get_image_path_from_input_folder()
    tissue_mask_path = get_tissue_mask_path_from_input_folder()

    print(f'Processing image: {image_path}')
    print(f'Processing with mask: {tissue_mask_path}')

    # open images
    image = open_multiresolutionimage_image(path=image_path)
    tissue_mask = open_multiresolutionimage_image(path=tissue_mask_path)

    dimensions = image.getDimensions()
    wsi_width, wsi_height = dimensions[0], dimensions[1]

    thumbnail_level = 5
    tissue_mask_thumb = tissue_mask.getUCharPatch(startX=0, startY=0, width=wsi_width // (2**thumbnail_level), height=wsi_height // (2**thumbnail_level),
                                                  level=thumbnail_level).squeeze()
    print('tissue_mask_thumb shape:', tissue_mask_thumb.shape)
    all_foreground_num = np.sum(tissue_mask_thumb)
    print('all_foreground_num:', all_foreground_num)

    # get image info

    spacing = image.getSpacing()
    print(f'dimensions: {dimensions}')
    print(f'spacing: {spacing}')

    # create writers
    print(f"Setting up writers")
    segmentation_writer = SegmentationWriter(
        TMP_SEGMENTATION_OUTPUT_PATH,
        tile_size=tile_size,
        dimensions=dimensions,
        spacing=spacing,
    )
    detection_writer = DetectionWriter(TMP_DETECTION_OUTPUT_PATH)
    tils_score_writer = TilsScoreWriter(TMP_TILS_SCORE_PATH)

    print("Processing image...")
    # loop over image and get tiles
    record_dict = {}  # record the the number of pixels of each category for each patch

    """
    set the seg params, get the seg dataloader, and input the batch of images to the seg net
    """

    seg_scale, foreground_thresh, tumor_num_thresh = get_thresh_params(all_foreground_num)

    # loop over the tissue mask, and get the regions which will be input to the seg net.
    # The regions with the number of foreground less than the foreground_thresh will not be input to the seg net
    ts = time.time()
    print('##Loop over the tissue mask, and get the coordinates of the regions which will be input to the seg net.##')
    print('##The regions with the number of foreground less than the foreground_thresh will not be input to the seg net##')
    input_seg_net_regions_lst, segmentation_writer = get_all_regions_seg(dimensions, tile_size, foreground_thresh, tissue_mask, segmentation_writer)
    te = time.time()
    print("###Loop over the tissue mask, and get the regions of segmentation took: %2.4f sec" % (te - ts))

    ts = time.time()
    print('##Segmentation start##')
    seg_transform = transforms.Compose([
        transforms.Resize((int(tile_size * seg_scale), int(tile_size * seg_scale))),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    seg_batch_loader = get_seg_loaders(input_seg_net_regions_lst, seg_transform, image, batchsize=6)
    det_regions_lst = []
    # input the batch of images to the seg net
    for data in tqdm(seg_batch_loader):
        with torch.no_grad():
            seg_batch_img, info = data[0].cuda(), data[1]
            info = info.data.numpy()
            batch_size = seg_batch_img.size(0)
            seg_output_batch = seg_net(seg_batch_img)
            seg_output_up = torch.nn.functional.interpolate(seg_output_batch,
                                                            size=(256, 256), mode='bilinear',
                                                            align_corners=True)
            # print('seg_output_up size:', seg_output_up.size())
            _, pred = torch.max(seg_output_up, 1)

            for n in range(0, batch_size):
                per_pred = pred[n].data.cpu().numpy()
                segmentation_mask = per_pred + 1
                DCI_num, healthy_grand_num, tumor_num, inflame_s_num, all_num = get_seg_num(segmentation_mask)
                record_dict[info[n]] = [DCI_num, healthy_grand_num, tumor_num, inflame_s_num, all_num]
                if tumor_num < tumor_num_thresh:
                    detections = []
                    lymcells_num_256 = inflame_s_num + 150
                    record_dict[info[n]].append(lymcells_num_256)
                else:
                    det_regions_lst.append(info[n])

                segmentation_img = Image.fromarray(segmentation_mask.astype(np.uint8))
                segmentation_img = transforms.Resize(size=(tile_size, tile_size), interpolation=Image.NEAREST)(
                    segmentation_img)
                segmentation_mask_2048 = np.array(segmentation_img)
                segmentation_writer.write_segmentation(tile=segmentation_mask_2048, x=input_seg_net_regions_lst[info[n]][0], y=input_seg_net_regions_lst[info[n]][1])
    te = time.time()
    print("###The segmentation process took: %2.4f sec" % (te - ts))

    """
    set the detection params, get the det dataloader, and input the batch of images to the det net
    """
    ts = time.time()
    print('##Detection start##')
    detect_transform = transforms.Compose([
        transforms.Resize((int((tile_size // 4) * 3), int((tile_size // 4) * 3))),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    det_batch_loader = get_det_loaders(det_regions_lst, input_seg_net_regions_lst, detect_transform, image, batchsize=6)
    # input the batch of images to the det net
    for data in tqdm(det_batch_loader):
        with torch.no_grad():
            det_batch_img, info = data[0].cuda(), data[1]
            info = info.data.numpy()
            batch_size = det_batch_img.size(0)
            detect_output = det_net(det_batch_img)
            # print('detect_output size:', detect_output.size())
            detect_output = detect_output.data.cpu().numpy()
            for n in range(0, batch_size):
                per_det_output = detect_output[n, 0]
                per_det_output = np.maximum(0, per_det_output)
                per_det_output = np.minimum(1, per_det_output)
                per_det_output = (per_det_output * 255).astype(np.uint8)
                per_det_output = Image.fromarray(per_det_output)
                det_img_256 = transforms.Resize(size=(256, 256))(per_det_output)
                det_img_array_256 = np.array(det_img_256)
                points_idx = np.where(det_img_array_256 > 3)
                lymcells_num_256 = len(points_idx[0])
                if lymcells_num_256 == 0:
                    detections = []
                else:
                    ys, xs = points_idx[0] * 8, points_idx[1] * 8
                    probabilities = det_img_array_256[points_idx[0], points_idx[1]]
                    probabilities = probabilities / 255
                    detections = list(zip(xs, ys, probabilities))
                record_dict[info[n]].append(lymcells_num_256)
                detection_writer.write_detections(
                    detections=detections, spacing=spacing, x_offset=input_seg_net_regions_lst[info[n]][0], y_offset=input_seg_net_regions_lst[info[n]][1]
                )
    te = time.time()
    print("###The detection process took: %2.4f sec" % (te - ts))

    '''
    cal the til score for each patch
    '''
    ts = time.time()
    print('##TIL computation start##')
    iot_lst = []  # cal til based on inlfame stroma
    lot_lst = []  # cal til based on lym cell
    tumor_lst = []
    inflame_lst = []
    lym_lst = []
    for per_key in record_dict:
        DCI_num, healthy_grand_num, tumor_num, inflame_s_num, all_num, lymcells_num_256 = record_dict[per_key]
        tumor_lst.append(tumor_num)
        inflame_lst.append(inflame_s_num)
        lym_lst.append(lymcells_num_256)

        DCI_hg_num = DCI_num + healthy_grand_num
        DCI_hg_of_all = DCI_hg_num / (256 * 256)

        base_num_inflame = 10000
        base_num_lymcell = 13500
        base_num = 11000
        """the number of inflame stroma and lym cells is divided by the corresponding base num. 
           Since the seg model may recognize the DCI or healthy grand as invasive tumor, the til will be set lower once 
           DCI or healthy grand are recognized"""
        inflame_of_tumor = (1 - DCI_hg_of_all) * inflame_s_num / base_num_inflame
        lymcell_of_tumor = (1 - DCI_hg_of_all) * lymcells_num_256 / base_num_lymcell

        if DCI_hg_of_all > 0.01:
            inflame_of_tumor = inflame_of_tumor * 0.9
            lymcell_of_tumor = lymcell_of_tumor * 0.9

        if tumor_num >= base_num:
            inflame_of_tumor = inflame_of_tumor * (tumor_num / base_num)
            lymcell_of_tumor = lymcell_of_tumor * (tumor_num / base_num)
            iot_lst.append(inflame_of_tumor)
            lot_lst.append(lymcell_of_tumor)
        elif (tumor_num < base_num) and (tumor_num >= 3000):  # 2621 3000 4000
            iot_lst.append(inflame_of_tumor)
            lot_lst.append(lymcell_of_tumor)

    '''cal the til score for the entire wsi'''
    if len(iot_lst) == 0:
        tumor_total = sum(tumor_lst)
        inflame_total = sum(inflame_lst)
        lym_total = sum(lym_lst)
        if tumor_total == 0:
            iot_lst.append(0)
            lot_lst.append(0)
        else:
            iot_lst.append(min(0.5, inflame_total / tumor_total))
            lot_lst.append(min(0.5, lym_total / tumor_total))

    iot_mean = sum(iot_lst) / len(iot_lst)
    lot_mean = sum(lot_lst) / len(lot_lst)
    tils_score = (iot_mean + lot_mean + lot_mean) / 3
    if tils_score >= 0.95:
        tils_score = 0.95
    tils_score = int(tils_score * 100)
    tils_score = min(100, tils_score)
    tils_score = max(0, tils_score)
    te = time.time()
    print("###The til computation took: %2.4f sec" % (te - ts))

    print("Saving...")
    # save segmentation and detection
    segmentation_writer.save()
    detection_writer.save()

    print('Number of detections', len(detection_writer.detections))

    print("Compute tils score...")
    print('tils_score:', tils_score)
    tils_score_writer.set_tils_score(tils_score=tils_score)

    print("Saving...")
    # save tils score
    tils_score_writer.save()

    print("Copy data...")
    # save data to output folder
    copy_data_to_output_folders()

    print("Completed!")
