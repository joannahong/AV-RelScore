import cv2
import os
import albumentations as A
import numpy as np
import random
from skimage.util import random_noise
import torchvision
import torch
import pickle
random.seed(10)
np.random.seed(10)
_window_margin = 12

def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    """Overlay `img_overlay` onto `img` at (x, y) and blend using `alpha_mask`.

    `alpha_mask` must have same HxW as `img_overlay` and values in range [0, 1].
    """
    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha
    img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop

    return img

def get_occluder_augmentor():
    """
    Occludor augmentor
    """
    aug=A.Compose([
        A.AdvancedBlur(),
        A.OneOf([
            A.ImageCompression (quality_lower=70,p=0.5),
            ], p=0.5),
        A.Affine  (
            scale=(0.8,1.2),
            rotate=(-15,15),
            shear=(-8,8),
            fit_output=True,
            p=0.7
        ),
        A.RandomBrightnessContrast(p=0.5,brightness_limit=0.1, contrast_limit=0.1, brightness_by_max=False)
        ])
    return aug

def get_occluders(d, d_mask, data='LRS2'):
    aug = get_occluder_augmentor()

    size = 30 if data == 'LRS2' else 42

    occlude_imgs = os.listdir(d)
    occlude_img = random.choice(occlude_imgs)
    occlude_mask = occlude_img.replace('jpeg', 'png')

    ori_occluder_img = cv2.imread(os.path.join(d, occlude_img), -1)
    try:
        ori_occluder_img = cv2.cvtColor(ori_occluder_img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(e)
        exit()

    occluder_mask = cv2.imread(os.path.join(d_mask, occlude_mask))
    occluder_mask = cv2.cvtColor(occluder_mask, cv2.COLOR_BGR2GRAY)

    occluder_mask = cv2.resize(occluder_mask, (ori_occluder_img.shape[1], ori_occluder_img.shape[0]), interpolation=cv2.INTER_LANCZOS4)

    try:
        occluder_img = cv2.bitwise_and(ori_occluder_img, ori_occluder_img, mask=occluder_mask)
    except Exception as e:
        print(e)
        return

    transformed = aug(image=occluder_img, mask=occluder_mask)
    occluder_img, occluder_mask = transformed["image"], transformed["mask"]

    occluder_img = cv2.resize(occluder_img, (size,size), interpolation= cv2.INTER_LANCZOS4)
    occluder_mask = cv2.resize(occluder_mask, (size,size), interpolation= cv2.INTER_LANCZOS4)

    return occlude_img, occluder_img, occluder_mask

def occlude_sequence(d, d_mask, img_seq, landmarks, freq=1, bgr=False, data='LRS2'):
    if freq == 1:
        occlude_img, occluder_img, occluder_mask = get_occluders(d, d_mask, data=data)
        len = img_seq.shape[0]
        start_pt_idx = random.randint(48,67)
        offset_x = 15
        offset_y = 15
        occ_len = random.randint(int(len * 0.3), int(len * 0.5))
        start_fr = random.randint(0, len-occ_len)

        for i in range(occ_len):
            fr = img_seq[i+start_fr]
            x, y = landmarks[i,start_pt_idx]
            alpha_mask = np.expand_dims(occluder_mask, axis=2)
            alpha_mask = np.repeat(alpha_mask, 3, axis=2) / 255.0
            fr = overlay_image_alpha(fr, occluder_img, int(x-offset_x), int(y-offset_y), alpha_mask)
            img_seq[i + start_fr] = fr
    else:
        len_global = img_seq.shape[0]
        len = img_seq.shape[0] // freq
        for j in range(freq):
            occlude_img, occluder_img, occluder_mask = get_occluders(d, d_mask, data=data)
            start_pt_idx = random.randint(48, 67)
            offset_x = 15
            offset_y = 15
            try:
                occ_len = random.randint(int(len_global * 0.3), int(len_global * 0.5))
                start_fr = random.randint(0, len*j + len - occ_len)
                if start_fr < len*j:
                    assert 1==2
            except:
                occ_len = len // 2
                start_fr = len * j

            for i in range(occ_len):
                fr = img_seq[i + start_fr]
                x, y = landmarks[i, start_pt_idx]
                alpha_mask = np.expand_dims(occluder_mask, axis=2)
                alpha_mask = np.repeat(alpha_mask, 3, axis=2) / 255.0
                fr = overlay_image_alpha(fr, occluder_img, int(x-offset_x), int(y-offset_y), alpha_mask)
            img_seq[i + start_fr] = fr
    if bgr:
        temp_imgs = []
        for im in img_seq:
            temp_imgs.append(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        img_seq = temp_imgs

    return np.array(img_seq), occlude_img

def occlude_sequence_noise(img_seq, freq=1):

    if freq == 1:
        len = img_seq.shape[0]
        occ_len = random.randint(int(len * 0.1), int(len * 0.5))
        start_fr = random.randint(0, len-occ_len)

        raw_sequence = img_seq[start_fr:start_fr+occ_len]
        prob = random.random()
        if prob < 0.5:
            var = random.random() * 0.2
            raw_sequence = random_noise(raw_sequence, mode='gaussian', mean=0, var=var, clip=True) * 255
        elif prob < 1.0:
            blur = torchvision.transforms.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 2.0))
            raw_sequence = blur(torch.tensor(raw_sequence).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).numpy()
        else:
            pass

        img_seq[start_fr:start_fr + occ_len] = raw_sequence

    else:
        len_global = img_seq.shape[0]
        len = img_seq.shape[0] // freq
        for j in range(freq):
            try:
                occ_len = random.randint(int(len_global * 0.3), int(len_global * 0.5))
                start_fr = random.randint(0, len*j + len - occ_len)
                if start_fr < len*j:
                    assert 1==2
            except:
                occ_len = len // 2
                start_fr = len * j

            raw_sequence = img_seq[start_fr:start_fr + occ_len]
            prob = random.random()
            if prob < 0.5:
                var = random.random() * 0.2
                raw_sequence = random_noise(raw_sequence, mode='gaussian', mean=0, var=var, clip=True) * 255
            elif prob < 1.0:
                blur = torchvision.transforms.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 2.0))
                raw_sequence = blur(torch.tensor(raw_sequence).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).numpy()
            else:
                pass

            img_seq[start_fr:start_fr + occ_len] = raw_sequence
    temp_imgs = []
    for im in img_seq:
        temp_imgs.append(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    return np.array(temp_imgs)

def landmarks_interpolate(landmarks):
    """landmarks_interpolate.

    :param landmarks: List, the raw landmark (in-place)

    """
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    if not valid_frames_idx:
        return None
    for idx in range(1, len(valid_frames_idx)):
        if valid_frames_idx[idx] - valid_frames_idx[idx - 1] == 1:
            continue
        else:
            landmarks = linear_interpolate(landmarks, valid_frames_idx[idx - 1], valid_frames_idx[idx])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    # -- Corner case: keep frames at the beginning or at the end failed to be detected.
    if valid_frames_idx:
        landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
        landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"
    return landmarks


def crop_patch(video_pathname, landmarks):
    """crop_patch.

    :param video_pathname: str, the filename for the processed video.
    :param landmarks: List, the interpolated landmarks.
    """
    frame_idx = 0
    frame_gen = read_video(video_pathname)
    while True:
        try:
            frame = frame_gen.__next__()  ## -- BGR
        except StopIteration:
            break
        if frame_idx == 0:
            sequence = []
        sequence.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_idx += 1
    return np.array(sequence)


def preprocess(video_pathname, landmarks_pathname):
    """preprocess.

    :param video_pathname: str, the filename for the video.
    :param landmarks_pathname: str, the filename for the landmarks.
    """
    # -- Step 1, extract landmarks from pkl files.
    if isinstance(landmarks_pathname, str):
        with open(landmarks_pathname, "rb") as pkl_file:
            landmarks = pickle.load(pkl_file)
    else:
        landmarks = landmarks_pathname
    # -- Step 2, pre-process landmarks: interpolate frames that not being detected.
    preprocessed_landmarks = landmarks_interpolate(landmarks)
    # -- Step 3, exclude corner cases:
    #   -- 1) no landmark in all frames
    #   -- 2) number of frames is less than window length.
    if not preprocessed_landmarks or len(preprocessed_landmarks) < _window_margin:
        return None, None, None, None
    # -- Step 4, affine transformation and crop patch
    sequence = crop_patch(video_pathname, preprocessed_landmarks)

    assert sequence is not None, "cannot crop from {}.".format(video_pathname)
    return sequence, np.array(preprocessed_landmarks)


def load_video(data_filename, landmarks_filename=None):
    """load_video.

    :param data_filename: str, the filename of input sequence.
    :param landmarks_filename: str, the filename of landmarks.
    """
    assert landmarks_filename is not None
    sequence, landmark = preprocess(
        video_pathname=data_filename,
        landmarks_pathname=landmarks_filename,
    )
    return sequence, landmark


def read_video(filename):
    """load_video.

    :param filename: str, the fileanme for a video sequence.
    """
    cap = cv2.VideoCapture(filename)
    while (cap.isOpened()):
        ret, frame = cap.read()  # BGR
        if ret:
            yield frame
        else:
            break
    cap.release()


def write_video(video, filename):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter(filename, fourcc, 25, (video.shape[1], video.shape[2]))
    for i, frame in enumerate(video):
        output.write(frame)
    output.release()


def linear_interpolate(landmarks, start_idx, stop_idx):
    """linear_interpolate.

    :param landmarks: ndarray, input landmarks to be interpolated.
    :param start_idx: int, the start index for linear interpolation.
    :param stop_idx: int, the stop for linear interpolation.
    """
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx - start_idx):
        landmarks[start_idx + idx] = start_landmarks + idx / float(stop_idx - start_idx) * delta
    return landmarks
