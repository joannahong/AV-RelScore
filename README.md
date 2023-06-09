# AV-RelScore
This code is part of the paper: [**Watch or Listen: Robust Audio-Visual Speech Recognition with Visual Corruption Modeling and Reliability Scoring**](https://arxiv.org/pdf/2303.08536.pdf) accepted at CVPR 2023.

## Overview
This repository provides the audio-visual corruption modeling code for testing audio-visual speech recognition of LRS2 and LRS3 datasets.
The video demo is available in [here](https://github.com/joannahong/AV-RelScore/tree/main/demo_video).

## Prerequisite
1. Python >= 3.6
2. Clone this repository.
3. Install python requirements. 
```
pip install -r requirements.txt
```
4. Download the [LRS2-BBC](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html) and [LRS3-TED](https://mmai.io/datasets/lip_reading/) datasets.
5. Download the landmarks of LRS2 and LRS3 from [this repository](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages).
6. Download `coco_object.7z` from [here](https://drive.google.com/drive/folders/15nZETWlGMdcKY6aHbchRsWkUI42KTNs5), extract, and put `object_image_sr` and `object_mask_x4` in `occlusion_patch` folder.

## Audio-Visual corruption modeling
- We utilize babble noise from [NOISEX-92](http://svr-www.eng.cam.ac.uk/comp.speech/Section1/Data/noisex.html) for the audio corruption modeling.
- The occlusion patches for the visual corruption modeling are provided from [this paper](https://arxiv.org/abs/2205.06218). 
- Please create the separate audio (.wav) files from the LRS2 and LRS3 video dataset.

### Audio corruption modeling
* LRS2

```Shell
python LRS2_audio_gen.py --split_file <SPLIT-FILENAME-PATH> \
                         --LRS2_main_dir <DATA-DIRECTORY-PATH> \
                         --LRS2_save_loc <OUTPUT-DIRECTORY-PATH> \
                         --babble_noise <BABBLE-NOISE-LOCATION> \
```

* LRS3

```Shell
python LRS3_audio_gen.py --split_file <SPLIT-FILENAME-PATH> \
                         --LRS3_test_dir <DATA-DIRECTORY-PATH> \
                         --LRS3_save_loc <OUTPUT-DIRECTORY-PATH> \
                         --babble_noise <BABBLE-NOISE-LOCATION> \
```

### Visual corruption modeling
* LRS2
```Shell
python LRS2_gen.py --split_file <SPLIT-FILENAME-PATH> \
                   --LRS2_main_dir <DATA-DIRECTORY-PATH> \
                   --LRS2_landmark_dir <LANDMARK-DIRECTORY-PATH> \
                   --LRS2_save_loc <OUTPUT-DIRECTORY-PATH> \
                   --occlusion <OCCLUSION-LOCATION> \
                   --occlusion_mask <OCCLUSION-MASK-LOCATION> \
```
* LRS3
```Shell
python LRS3_gen.py --split_file <SPLIT-FILENAME-PATH> \
                   --LRS3_test_dir <DATA-DIRECTORY-PATH> \
                   --LRS3_landmark_dir <LANDMARK-DIRECTORY-PATH> \
                   --LRS3_save_loc <OUTPUT-DIRECTORY-PATH> \
                   --occlusion <OCCLUSION-LOCATION> \
                   --occlusion_mask <OCCLUSION-MASK-LOCATION> \
```
## Test datasets
Note that the extracted corrupted data may be different from the actual corrupted test datasets that we have used for the experiment. Since we use *random* function when modeling the audio-visual corruption, so it may not work the same on all devices.

Please request us (joanna2587@kaist.ac.kr) the actual test datasets for the fair comparisons.

## Acknowledgement
We refer to [Visual Speech Recognition for Multiple Languages](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages) for landmarks of the datasets and [Delving into High-Quality Synthetic Face Occlusion Segmentation Datasets](https://github.com/kennyvoo/face-occlusion-generation) for visual occlusion patches. We thank the authors for the amazing works.


## Citation
If you find our AV-RelSocre useful in your research, please cite our [paper](https://arxiv.org/abs/2303.08536).
```BibTeX
@article{hong2023watch,
  title={Watch or Listen: Robust Audio-Visual Speech Recognition with Visual Corruption Modeling and Reliability Scoring},
  author={Hong, Joanna and Kim, Minsu and Choi, Jeongsoo and Ro, Yong Man},
  journal={arXiv preprint arXiv:2303.08536},
  year={2023}
}

