# Algorithm for Tiger Challenge

This repository includes the codes and the dockerfile for computing TILs given a H&E breast cancer histopathology slide. 
The descriptions of the algorithm can be found [here] (https://tiger.grand-challenge.org/evaluation/survival-final-evaluation/leaderboard/) with the user 大胖胖墩(Xulin Chen).

## Run the algotithm
- The model weights of this repository can be obtained [here] (https://drive.google.com/file/d/11On7kDKU79ubP00_jCIvEJMnVS_vgpzd/view?usp=sharing). 
After downloading it, please put the two files within the folder to the ./model_weight/ of this repository. 
The model weights are on the CC BY-NC 4.0 license.
- The framework of the inference process is as following. The tissue mask is looped over firstly to get the 
coordinates of regions for segmentation. Then the dataloader of torch is used to input batch of images to the seg net.
Based on the results of segmentation, the target regions for detection are collected. Then the batch of images is 
input to the detection network by dataloader. The number of pixels of each category is recorded for each patch, since I 
calculate the til score based on the patches. 
- In the "test.sh", the shm-size can be set larger, and the num_workers of DataLoader can then be set higher. 
- In the "test.sh", the "--runtime nvidia" is added to "docker run". This may not need to be added, and it seems to depend on your environment.

## Requirements

- Ubuntu software
  - Ubuntu 20.04
  - ASAP 2.1
  
- Python packages
  - tqdm 4.62.3
  - Pytorch 1.9.0

## Authors
This code is made by Xulin Chen. It is based on the code developed by the TIGER challenge organizers.

