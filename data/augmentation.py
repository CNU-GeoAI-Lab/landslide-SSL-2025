import math
import numpy as np
import functools
import tensorflow as tf
import tensorflow_addons as tfa

from copy import deepcopy
from scipy.ndimage import gaussian_filter

def random_crop(image, seed=None):
    """
    image: (28, 28, 20) 크기의 입력 텐서
    crop_size: 목표 크기 (예: 18x18x20)
    """
    # image = np.array(image)
    crop_size = (int(image.shape[1]/3*2), int(image.shape[1]/3*2), image.shape[2])
    image = tf.image.random_crop(image, size=crop_size, seed=seed)
    # center_x = np.random.randint(11, 16, 1)
    # center_y = np.random.randint(11, 16, 1)
    # crop_image = image[int(center_x[0])-9: int(center_x[0])+9, int(center_y[0])-9:int(center_y[0])+9, :]
    # crop_image = tf.convert_to_tensor(crop_image)
    return image
    # return crop_image

# 랜덤 좌우 플립
def random_flip(image, seed=None):
    # prob = tf.random.uniform([1], 0, 1)
    # if prob>0.5:
    flip = tf.image.random_flip_left_right(image, seed=seed)
    # else:
    #     flip = tf.image.random_flip_up_down(image, seed=seed)
    return flip

# 랜덤 회전 (90도 단위)
def random_rotation(image, seed=None):
    k = tf.random.uniform(shape=(), minval=-1, maxval=1, dtype=tf.int32, seed=seed)
    return tfa.image.rotate(image, tf.cast(k, tf.float32) * np.pi / 2)

# 가우시안 노이즈 추가
def gaussian_noise(image):
    sigma = np.random.uniform(0, 0.4, [1])
    blurred = gaussian_filter(image, sigma=sigma[0])
    return blurred

# 채널별 밝기 조정 (RGB 대신 전체 채널에 적용)
def random_brightness(image, max_delta=0.1, seed=None):
    return tf.image.random_brightness(image, max_delta=max_delta, seed=seed)

def channel_drop(image, drop_features=2, seed=None):
    """
    입력 이미지에서 일부 채널을 무작위로 제거합니다.
    image: (H, W, C) 텐서 (예: 12x12x20)
    drop_features: 제거할 채널 수
    """
    image = np.array(image)
    if seed is not None:
        np.random.seed(seed)

    drop_features_num = np.random.randint(0, 20, drop_features)
    image[:, :, drop_features_num] = 0  # 선택된 채널을 0으로 설정

    return image

def random_mask_except_center(image, mask_value=0., p=0.5):
    """
    입력 이미지에서 중앙 영역을 제외한 나머지에 임의의 마스킹을 적용합니다.
    image: (H, W, C) 텐서 (예: 12x12x20)
    mask_value: 마스킹 값 (기본 0)
    """
    h, w, c = image.shape
    center_h, center_w = 6, 6
    start_h = (h - center_h) // 2
    start_w = (w - center_w) // 2

    # 전체 마스크 생성 (True: 마스킹, False: 보존)
    mask = np.ones((h, w, c), dtype=bool)
    mask[start_h:start_h+center_h, start_w:start_w+center_w] = image[start_h:start_h+center_h, start_w:start_w+center_w]

    # 임의로 마스킹할 위치 선택 (중앙 4x4는 제외)
    random_mask = np.random.uniform(0, 1, size = (h, w)) # p 확률로 마스킹
    random_mask[np.where(random_mask < p)] = 0
    random_mask[np.where(random_mask >= p)] = 1

    for i in range(c):
        image[:, :, i]*random_mask  # 마스킹 값으로 설정

    image[start_h:start_h+center_h, start_w:start_w+center_w] = mask[start_h:start_h+center_h, start_w:start_w+center_w]
    return image

# SimCLR 스타일의 증강 파이프라인
def data_augmentation(image, aug, seed=None):
    """
    image: (12, 12, 20) 크기의 입력 이미지
    crop_size: 크롭 후 목표 크기
    """
    # 입력 이미지가 0~1 사이 값이라고 가정
    # image = tf.ensure_shape(image, [12, 12, 20])
    # crop_size = image.shape
    # 1. 랜덤 크롭
    image = random_crop(image, seed=seed)
    
    if aug[0]==1:
        image = random_flip(image, seed=seed)
    
    if aug[1]==1:
        image = random_rotation(image, seed=seed)
    
    if aug[2]==1:
        image = random_brightness(image, max_delta=0.1, seed=seed)
    
    if aug[3]==1:
        image = gaussian_noise(image)
    
    image = np.array(image)
    if aug[4]==1:
        image = random_mask_except_center(image, mask_value=0., p=0.3)

    return image

def data_augmentation_w_coors(image, aug, seed=None):
    """
    image: (12, 12, 20) 크기의 입력 이미지
    crop_size: 크롭 후 목표 크기
    """
    # 입력 이미지가 0~1 사이 값이라고 가정
    # image = tf.ensure_shape(image, [12, 12, 20])
    # crop_size = image.shape
    # 1. 랜덤 크롭
    image = random_crop(image, seed=seed)
    
    if aug[0]==1:
        image = random_flip(image, seed=seed)
    
    if aug[1]==1:
        image = random_rotation(image, seed=seed)
    
    coors = image[:,:,20:]
    image = image[:,:,:20]
    
    if aug[2]==1:
        image = random_brightness(image, max_delta=0.1, seed=seed)
    
    if aug[3]==1:
        image = gaussian_noise(image)
    
    image = np.array(image)
    if aug[4]==1:
        image = random_mask_except_center(image, mask_value=0., p=0.3)

    return np.concatenate([image, coors], axis=2)

def random_data_augmentation(image, seed=None, thresholds=[0.5, 0.5, 0.5, 0.6, 0.4]):
    """
    image: (12, 12, 20) 크기의 입력 이미지
    crop_size: 크롭 후 목표 크기
    thresholds: 각 augmentation에 대한 threshold 리스트 [flip, rotation, brightness, gaussian_noise, mask]
    """
    # 입력 이미지가 0~1 사이 값이라고 가정
    # image = tf.ensure_shape(image, [12, 12, 20])
    # crop_size = image.shape
    # 1. 랜덤 크롭
    image = random_crop(image, seed=seed)
    
    prob_ = []
    for _ in range(6):
        prob = np.random.uniform(0,1, [1])
        prob_.append(prob)

    if prob_[0] > thresholds[0]:
        image = random_flip(image, seed=seed)
    
    if prob_[1] > thresholds[1]:
        image = random_rotation(image, seed=seed)
    
    if prob_[2] > thresholds[2]:
        image = random_brightness(image, max_delta=0.1, seed=seed)
    
    if prob_[3] > thresholds[3]:
        image = gaussian_noise(image)

    image = np.array(image)
    if prob_[4] > thresholds[4]:
        image = random_mask_except_center(image, mask_value=0.0001, p=0.3)

    # if prob_[5] > 0.5:
    #     image = channel_drop(image, drop_features=1, seed=seed)

    return image

def random_data_augmentation_w_coors(image, seed=None, thresholds=[0.5, 0.5, 0.5, 0.5, 0.5]):
    """
    image: (12, 12, 20) 크기의 입력 이미지
    crop_size: 크롭 후 목표 크기
    thresholds: 각 augmentation에 대한 threshold 리스트 [flip, rotation, brightness, gaussian_noise, mask]
    """
    # 입력 이미지가 0~1 사이 값이라고 가정
    # image = tf.ensure_shape(image, [12, 12, 20])
    # crop_size = image.shape
    # 1. 랜덤 크롭
    image = random_crop(image, seed=seed)
    
    prob_ = []
    for _ in range(6):
        prob = np.random.uniform(0,1, [1])
        prob_.append(prob)

    if prob_[0] > thresholds[0]:
        image = random_flip(image, seed=seed)
    
    if prob_[1] > thresholds[1]:
        image = random_rotation(image, seed=seed)

    coors = image[:,:,20:]
    image = image[:,:,:20]
    
    if prob_[2] > thresholds[2]:
        image = random_brightness(image, max_delta=0.1, seed=seed)
    
    if prob_[3] > thresholds[3]:
        image = gaussian_noise(image)

    image = np.array(image)
    if prob_[4] > thresholds[4]:
        image = random_mask_except_center(image, mask_value=0.0001, p=0.3)

    # if prob_[5] > 0.5:
    #     image = channel_drop(image, drop_features=1, seed=seed)

    return np.concatenate([image, coors], axis=2)

def apply_simclr_augmentation(image, aug_1, aug_2, random = False, random_aug_thresholds=[0.5, 0.5, 0.5, 0.6, 0.4]):
    # 두 개의 서로 다른 증강된 뷰 생성 (SimCLR의 핵심)
    #seed1 = tf.random.uniform([1], maxval=1000, dtype=tf.int32)
    #seed2 = tf.random.uniform([1], maxval=1000, dtype=tf.int32)
    
    if random:   
        view1 = random_data_augmentation(image, thresholds=random_aug_thresholds)
        view2 = random_data_augmentation(image, thresholds=random_aug_thresholds)
    else:
        view1 = data_augmentation(image, aug_1)
        view1 = data_augmentation(image, aug_2)

    return view1, view2

def apply_simclr_augmentation_w_coors(image, aug_1, aug_2, random = False, random_aug_thresholds=[0.5, 0.5, 0.5, 0.5, 0.5]):
    # 두 개의 서로 다른 증강된 뷰 생성 (SimCLR의 핵심)
    #seed1 = tf.random.uniform([1], maxval=1000, dtype=tf.int32)
    #seed2 = tf.random.uniform([1], maxval=1000, dtype=tf.int32)
    
    if random:   
        view1 = random_data_augmentation_w_coors(image, thresholds=random_aug_thresholds)
        view2 = random_data_augmentation_w_coors(image, thresholds=random_aug_thresholds)
    else:
        view1 = data_augmentation_w_coors(image, aug_1)
        view2 = data_augmentation_w_coors(image, aug_2)

    return view1, view2