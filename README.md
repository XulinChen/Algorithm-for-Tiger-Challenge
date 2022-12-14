# Algorithm for Tiger Challenge

This repository includes codes, model weights and docker for computing TILs given a H&E breast cancer histopathology slide. 
The descriptions of the algorithm can be found here https://tiger.grand-challenge.org/evaluation/survival-final-evaluation/leaderboard/ with the user 大胖胖墩(Xulin Chen).

## News
- The gpu has been added to the dockerfile.
- The ASAP has been changed to 2.1.
- The framework of the inference process has been changed a lot. The tissue mask is looped over firstly to get the 
coordinates of regions for segmentation. Then the dataloader of torch is used to input batch of images to the seg net.
Based on the results of segmentation, the target regions for detection are collected. Then the batch of images is 
input to the detection network by dataloader. The number of pixels of each category is recorded for each patch, since I 
calculate the til score based on the patches. 
- The time used for each stage can be seen.
- The total process will be 8 times faster than before.
- In the "test.sh", the shm-size can be set larger, and the num_workers of DataLoader can then set higher. 
- In the "test.sh", the "--runtime nvidia" is added to "docker run". This may not need to be add, and it seems to depend on your environment.

## Requirements

- Ubuntu software
  - Ubuntu20.04
  - ASAP 2.1
  
- Python packages
  - tqdm==4.62.3
  - Pytorch 1.9.0

## Authors
This code is made by Xulin Chen, a member at Cells Vision (Guangzhou) Medical Technology Inc, China. It is based on the code developed by the TIGER challenge organisers.

