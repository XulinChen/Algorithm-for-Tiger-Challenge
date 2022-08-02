from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm
from PIL import Image

from .seg_config import seg_config
from .seg_config import update_seg_config
from .seg_hrnet import get_seg_model

from .det_config import det_config
from .det_config import update_det_config
from .det_hrnet import get_cell_det_net
from .anchors import generate_default_anchor_maps, hard_nms
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


_, edge_anchors, _ = generate_default_anchor_maps(input_shape=(int(WRITING_TILE_SIZE), int(WRITING_TILE_SIZE)))

def process_image_tile_to_segmentation(
    image_tile: np.ndarray, tissue_mask_tile: np.ndarray, seg_model, seg_resize_scale
) -> np.ndarray:
    """Example function that shows processing a tile from a multiresolution image for segmentation purposes.
    
    NOTE 
        This code is only made for illustration and is not meant to be taken as valid processing step.

    Args:
        image_tile (np.ndarray): [description]
        tissue_mask_tile (np.ndarray): [description]

    Returns:
        np.ndarray: [description]
    """
    height, width = image_tile.shape[:2]
    imgs = Image.fromarray(image_tile)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    my_transform = transforms.Compose([
        transforms.Resize((int(height*seg_resize_scale), int(width*seg_resize_scale))),
        # transforms.Resize((height, width)),
        transforms.ToTensor(),
        normalize
    ])

    imgs = my_transform(imgs)
    imgs = imgs.unsqueeze(0)

    with torch.no_grad():
        seg_output = seg_model(imgs)
        seg_output_up = torch.nn.functional.interpolate(seg_output,
                                                        size=(256, 256), mode='bilinear',
                                                        align_corners=True)
        _, pred = torch.max(seg_output_up, 1)
        pred = pred.squeeze(0).numpy()
        pred = pred + 1

    prediction = np.copy(pred)
    return prediction * tissue_mask_tile


def process_image_tile_to_detections(
    image_tile: np.ndarray, segmentation_mask: np.ndarray, tissue_mask_tile: np.ndarray, det_model,
) -> List[tuple]:

    # if not np.any(segmentation_mask == 2):
        # return []

    height, width = image_tile.shape[:2]
    imgs = Image.fromarray(image_tile)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    detect_transform = transforms.Compose([
        transforms.Resize((int((height//4)*3), int((width//4)*3))),
        transforms.ToTensor(),
        normalize
    ])

    imgs = detect_transform(imgs)
    imgs = imgs.unsqueeze(0)

    with torch.no_grad():
        detect_output = det_model(imgs)
        det_output_array = detect_output.squeeze(0).squeeze(0).data.numpy()
        det_output_array = np.maximum(0, det_output_array)
        det_output_array = np.minimum(1, det_output_array)
        det_output_array = (det_output_array * 255).astype(np.uint8)
        det_img = Image.fromarray(det_output_array)
        det_img_256 = transforms.Resize(size=(tissue_mask_tile.shape[0], tissue_mask_tile.shape[1]))(det_img)
        det_img_array_256 = np.array(det_img_256)
        det_img_array_256 = det_img_array_256 * tissue_mask_tile
        points_idx = np.where(det_img_array_256 > 3)
        lymcells_num_256 = len(points_idx[0])
        if lymcells_num_256 == 0:
            return lymcells_num_256, []

        # det_output_array = detect_output.squeeze(0).squeeze(0).data.numpy()  # .data.cpu()
        # det_output_array = det_output_array * seg_mask_array
        # points_idx = np.where(det_output_array >= 0.68)
        ys, xs = points_idx[0] * 8, points_idx[1] * 8
        probabilities = det_img_array_256[points_idx[0], points_idx[1]]
        probabilities = probabilities / 255

    return lymcells_num_256, list(zip(xs, ys, probabilities))

def get_seg_num(seg_img_array_512):
    seg_pred_512 = np.copy(seg_img_array_512)

    tumor_num = len(np.where(seg_pred_512 == 1)[0])
    infame_s_num = len(np.where(seg_img_array_512 == 6)[0])
    all_num = seg_img_array_512.shape[0] * seg_img_array_512.shape[1]
    DCI_num = len(np.where(seg_img_array_512 == 3)[0])
    healthy_grand_num = len(np.where(seg_img_array_512 == 4)[0])
    return DCI_num, healthy_grand_num, tumor_num, infame_s_num, all_num

def process_til():
    """get the seg model"""
    config_path, checkpoint_path = Path('/home/usr/models/seg_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'), Path('/home/usr/models/tiger_seg_model_new_089_202204027.ckpt')
    # device = ",".join([str(ctx) for ctx in gpu_ctx])
    seg_ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    update_seg_config(seg_config, config_path)

    # ====loading model=====
    print('Loading seg model...')
    seg_net = get_seg_model(seg_config)
    seg_net.load_state_dict(seg_ckpt['net_state_dict'])
    seg_net.eval()

    '''get the detect model'''
    detect_config_path, detect_checkpoint_path = Path('/home/usr/models/lymcells_detect_tiger_hrnet_w18.yaml'), Path('/home/usr/models/lymcells_detect_256_384_768_768_079.ckpt')
    detect_ckpt = torch.load(detect_checkpoint_path, map_location=torch.device('cpu'))
    update_det_config(det_config, detect_config_path)
    print('Loading detection model...')
    det_net = get_cell_det_net(det_config)
    det_net.load_state_dict(detect_ckpt['net_state_dict'])
    det_net.eval()

    """Processes a test slide"""
    level = READING_LEVEL      # 0
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
    tissue_mask_thumb = tissue_mask.getUCharPatch(startX=0, startY=0, width=wsi_width//32, height=wsi_height//32, level=5).squeeze()
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
    iot_lst = []
    lot_lst = []
    tumor_lst = []
    infame_lst = []
    lym_lst = []

    '''
    These are some parameters for saving the processing time since the grand-challenge limit the time for each wsi.
    '''
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

    for y in tqdm(range(0, dimensions[1], tile_size)):
        for x in range(0, dimensions[0], tile_size):
            tissue_mask_tile = tissue_mask.getUCharPatch(
                startX=x, startY=y, width=tile_size, height=tile_size, level=level
            ).squeeze()
            tissue_mask_tile_img = Image.fromarray(tissue_mask_tile.astype(np.uint8))
            tissue_mask_256 = transforms.Resize(size=(tissue_mask_tile_img.size[0] // 8, tissue_mask_tile_img.size[1] // 8), interpolation=Image.NEAREST)(
                tissue_mask_tile_img)
            tissue_mask_256 = np.array(tissue_mask_256)
            foreground_num = np.sum(tissue_mask_256)
            
            if foreground_num <= foreground_thresh:
                segmentation_mask = np.zeros((tile_size, tile_size), dtype=int)
                detections = []
                lymcells_num_256 = 0
                DCI_num, healthy_grand_num, tumor_num, infame_s_num = 0, 0, 0, 0
            else:
                image_tile = image.getUCharPatch(
                    startX=x, startY=y, width=tile_size, height=tile_size, level=level
                )
                # segmentation
                segmentation_mask = process_image_tile_to_segmentation(
                    image_tile=image_tile, tissue_mask_tile=tissue_mask_256, seg_model=seg_net, seg_resize_scale=seg_scale
                )

                DCI_num, healthy_grand_num, tumor_num, infame_s_num, all_num = get_seg_num(segmentation_mask)
                if tumor_num < tumor_num_thresh:
                    detections = []
                    lymcells_num_256 = infame_s_num
                else:
                    lymcells_num_256, detections = process_image_tile_to_detections(
                        image_tile=image_tile, segmentation_mask=segmentation_mask, tissue_mask_tile=tissue_mask_256, det_model=det_net
                    )
            tumor_lst.append(tumor_num)
            infame_lst.append(infame_s_num)
            lym_lst.append(lymcells_num_256)

            DCI_hg_num = DCI_num + healthy_grand_num
            DCI_hg_of_all = DCI_hg_num / (256*256)

            base_num_infame = 10000
            base_num_lymcell = 13500
            base_num = 11000
            tumor_min = 3000

            # when DCI or healthy grand recogized, reduce the value of TIL score
            infame_of_tumor = (1 - DCI_hg_of_all) * infame_s_num / base_num_infame
            lymcell_of_tumor = (1 - DCI_hg_of_all) * lymcells_num_256 / base_num_lymcell

            if tumor_num >= base_num:
                infame_of_tumor = infame_of_tumor * (tumor_num / base_num)
                lymcell_of_tumor = lymcell_of_tumor * (tumor_num / base_num)
                iot_lst.append(infame_of_tumor)
                lot_lst.append(lymcell_of_tumor)
            elif (tumor_num < base_num) and (tumor_num >= tumor_min):
                iot_lst.append(infame_of_tumor)
                lot_lst.append(lymcell_of_tumor)

            segmentation_img = Image.fromarray(segmentation_mask.astype(np.uint8))
            segmentation_img = transforms.Resize(size=(tile_size, tile_size), interpolation=Image.NEAREST)(segmentation_img)
            segmentation_mask_2048 = np.array(segmentation_img)
            segmentation_writer.write_segmentation(tile=segmentation_mask_2048, x=x, y=y)

            detection_writer.write_detections(
                detections=detections, spacing=spacing, x_offset=x, y_offset=y
            )

    if len(iot_lst) == 0:
        tumor_total = sum(tumor_lst)
        infame_total = sum(infame_lst)
        lym_total = sum(lym_lst)
        if tumor_total == 0:
            iot_lst.append(0)
            lot_lst.append(0)
        else:
            iot_lst.append(min(0.5, infame_total / tumor_total))
            lot_lst.append(min(0.5, lym_total / tumor_total))

    iot_mean = sum(iot_lst) / len(iot_lst)
    lot_mean = sum(lot_lst) / len(lot_lst)
    tils_score = (iot_mean + lot_mean + lot_mean) / 3
    if tils_score >= 0.95:
        tils_score = 0.95
    tils_score = int(tils_score * 100)
    tils_score = min(100, tils_score)
    tils_score = max(0, tils_score)

    print("Saving...")
    # save segmentation and detection
    segmentation_writer.save()
    detection_writer.save()

    print('Number of detections', len(detection_writer.detections))
    
    print("Compute tils score...")
    # compute tils score

    print('tils_score:', tils_score)
    tils_score_writer.set_tils_score(tils_score=tils_score)

    print("Saving...")
    # save tils score
    tils_score_writer.save()

    print("Copy data...")
    # save data to output folder
    copy_data_to_output_folders()

    print("Completed!")
