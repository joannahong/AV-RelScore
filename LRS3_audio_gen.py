from Audio_perturb import *
import librosa
import soundfile as sf
import os
import argparse

random.seed(1)
def audio_gen(args):
    snrs = [-5, 0, 5, 10, 15]
    with open(args.split_file, 'r') as f:
        lines = f.readlines()
    test_list = []
    for l in lines:
        test_list.append(l.strip().split()[0])
    for snr in snrs:
        for kk, test_file in enumerate(test_list):
            f_name = os.path.join(args.LRS3_test_dir, test_file + '.wav')
            aud, _ = librosa.load(f_name, sr=16000)
            audio = noise_injection(aud, [args.babble_noise], snr=snr, part=True)
            if not os.path.exists(os.path.join(args.LRS3_save_loc + f'_{snr}', test_file.split('/')[0])):
                os.makedirs(os.path.join(args.LRS3_save_loc + f'_{snr}', test_file.split('/')[0]))
            save_name = os.path.join(args.LRS3_save_loc + f'_{snr}', test_file + '.wav')
            sf.write(save_name, audio, 16000)
            print(f'{snr}:{kk}/{len(test_list)}', end='\r')
        print('')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--split_file', type=str, default='./datasets/LRS3-TED/test.txt',
                        help='directory of test split file')
    parser.add_argument('-m', '--LRS3_test_dir', type=str, default='./datasets/LRS3-TED_audio/test',
                        required=False, help='directory of the original lrs2 audio dataset')
    parser.add_argument('-o', '--LRS3_save_loc', type=str, default='./datasets/LRS3-TED_audio/test_noise',
                        help='directory to save audio corruption files')
    parser.add_argument('--babble_noise', action='store_true', default='./noise/babble.wav',
                        help='load model weights only, ignore specified layers')

    args = parser.parse_args()
    audio_gen(args)