# AV-RelScore
This code is part of the paper: **Watch or Listen: Robust Audio-Visual Speech Recognition with Visual Corruption Modeling and Reliability Scoring** accepted at CVPR 2023.

## Overview
This repository provides the audio-visual corruption modeling code for testing audio-visual speech recognition of LRS2 and LRS3 datasets.

## Prerequisite
1. Python >= 3.6
2. Clone this repository.
3. Install python requirements. 
```
pip install -r requirements.txt
```
4. Download the [LRS2-BBC](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html) and [LRS3-TED](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html) datasets.
5. Download the landmarks of LRS2 and LRS3 from [this repository](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages).

## Audio-Visual corruption modeling

### Audio corruption modeling

### Visual corruption modeling

## Test datasets
Note that the extracted corrupted data may be different from the actual corrupted test datasets that we have used for the experiment. Since we use *random* function when modeling the audio-visual corruption, so it may not work the same on all devices.

Please request us (joanna2587@kaist.ac.kr) the actual test datasets for the fair comparisons.

<!--
## Citation
If you find our AV-RelSocre useful in your research, please cite below
```BibTeX
@article{shi2022avhubert,
    author  = {Bowen Shi and Wei-Ning Hsu and Kushal Lakhotia and Abdelrahman Mohamed},
    title = {Learning Audio-Visual Speech Representation by Masked Multimodal Cluster Prediction},
    journal = {arXiv preprint arXiv:2201.02184}
    year = {2022}
}
-->
