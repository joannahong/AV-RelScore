### audio noise gen ###
import numpy as np
import random
import os
import soundfile as sf
import librosa

def cal_adjusted_rms(clean_rms, snr):
    a = float(snr) / 20
    noise_rms = clean_rms / (10 ** a)
    return noise_rms

def cal_rms(amp):
    return np.sqrt(np.mean(np.square(amp), axis=-1))

def noise_injection(clean_amp, ns_list, snr=None, part=False):
    ns_file = random.choice(ns_list)

    noise_amp, _ = librosa.load(ns_file, sr=16000)

    if snr is None:
        snr = random.choice(snr_list)
    else:
        snr = snr

    if len(noise_amp) < len(clean_amp):
        num_repeat = (len(clean_amp) // len(noise_amp)) + 1
        noise_amp = np.concatenate([noise_amp] * num_repeat, 0)

    start = random.randint(0, len(noise_amp) - len(clean_amp))
    clean_rms = cal_rms(clean_amp)
    split_noise_amp = noise_amp[start: start + len(clean_amp)]
    noise_rms = cal_rms(split_noise_amp)
    adjusted_noise_rms = cal_adjusted_rms(clean_rms, snr)

    adjusted_noise_amp = split_noise_amp * (adjusted_noise_rms / noise_rms)
    if part:
        repeat = random.randint(1, 5)
        min_len = int(0.05 * 16000)
        max_len = int(len(adjusted_noise_amp) / repeat / 2)
        st_inds = random.sample(list(range(len(adjusted_noise_amp) - max_len + 1)), repeat)
        for st_ind in st_inds:
            adjusted_noise_amp[st_ind:st_ind + random.choice(list(range(min_len, max_len + 1)))] = 0.

    mixed_amp = (clean_amp + adjusted_noise_amp)
    if (mixed_amp.max(axis=0) > 1):
        mixed_amp = mixed_amp * (1 / mixed_amp.max(axis=0))
    return mixed_amp

