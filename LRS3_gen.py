from Visual_perturb import *
import random
import numpy as np
import os
import argparse
#
# split_file = '/mnt/hard/dataset/LRS3-TED/test.txt'
# LRS3_test_dir = '/mnt/hard/dataset/LRS3-TED/test'
# LRS3_landmark_dir = '/mnt/hard/dataset/LRS3_landmarks/test'
# LRS3_save_loc = '/mnt/hard/dataset/LRS3-TED/test3'
#
# d = '/mnt/hard2/Exp/Lip_reading/CV23/Interspeech_ESP/occlusion_patch/object_image_sr'
# d_mask = '/mnt/hard2/Exp/Lip_reading/CV23/Interspeech_ESP/occlusion_patch/object_mask_x4'

def video_gen(args):
    with open(args.split_file, 'r') as f:
        lines = f.readlines()
    test_list = []
    for l in lines:
        test_list.append(l.strip().split()[0])

    for mode in ['occlusion_and_noise', 'occlusion', 'noise']:
        random.seed(10)
        np.random.seed(10)
        save_dir = f'{args.LRS3_save_loc}_{mode}'
        for kk, test_file in enumerate(test_list):
            f_name = os.path.join(args.LRS3_test_dir, test_file + '.mp4')
            l_name = os.path.join(args.LRS3_landmark_dir, test_file + '.pkl')

            sequence, landmarks = load_video(f_name, l_name)
            if mode != 'noise':
                freq = random.choice([1, 2, 3])
                bgr = True if mode == 'occlusion' else False
                sequence, _ = occlude_sequence(args.occlusion, args.occlusion_mask, sequence, landmarks, freq=freq, bgr=bgr, data='LRS3')
            if mode != 'occlusion':
                freq = random.choice([1, 2, 3])
                sequence = occlude_sequence_noise(sequence, freq=freq)
            if not os.path.exists(os.path.join(save_dir, test_file.split('/')[0])):
                os.makedirs(os.path.join(save_dir, test_file.split('/')[0]))
            write_video(sequence, os.path.join(save_dir, test_file + '.mp4'))
            print(f'{mode}:{kk}/{len(test_list)}', end='\r')
        print('')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--split_file', type=str, default='./datasets/LRS3-TED/test.txt',
                        help='directory of test split file')
    parser.add_argument('-m', '--LRS3_test_dir', type=str, default='./datasets/LRS3-TED/test',
                        required=False, help='directory of the original lrs2 audio dataset')
    parser.add_argument('-l', '--LRS3_landmark_dir', type=str, default='./datasets/LRS3_landmarks/test',
                        required=False, help='directory of the original lrs2 audio dataset')
    parser.add_argument('-o', '--LRS3_save_loc', type=str, default='./datasets/LRS3-BBC/test3',
                        help='directory to save audio corruption files')
    parser.add_argument('--occlusion', action='store_true', default='./occlusion_patch/object_image_sr',
                        help='location of occlusion patch')
    parser.add_argument('--occlusion_mask', action='store_true', default='./occlusion_patch/object_mask_x4',
                        help='location of occlusion patch mask')

    args = parser.parse_args()
    video_gen(args)
