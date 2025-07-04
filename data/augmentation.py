import math
import numpy as np
import functools
import tensorflow as tf
import tensorflow_addons as tfa

from copy import deepcopy
from scipy.ndimage import gaussian_filter

def random_crop(image, seed=None):
    """
    image: (12, 12, 20) 크기의 입력 텐서
    crop_size: 목표 크기 (예: 8x8x20)
    """
    crop_size = (int(image.shape[1]/3*2), int(image.shape[1]/3*2), image.shape[2])
    return tf.image.random_crop(image, size=crop_size, seed=seed)

# 랜덤 좌우 플립
def random_flip(image, seed=None):
    prob = tf.random.uniform([1], 0, 1)
    if prob>0.5:
        flip = tf.image.random_flip_left_right(image, seed=seed)
    else:
        flip = tf.image.random_flip_up_down(image, seed=seed)
    return flip

# 랜덤 회전 (90도 단위)
def random_rotation(image, seed=None):
    k = tf.random.uniform(shape=(), minval=0, maxval=4, dtype=tf.int32, seed=seed)
    return tfa.image.rotate(image, tf.cast(k, tf.float32) * np.pi / 2)

# 가우시안 노이즈 추가
def gaussian_noise(image):
    sigma = np.random.uniform(0, 0.4, [1])
    blurred = gaussian_filter(image, sigma=sigma[0])
    return blurred

# 채널별 밝기 조정 (RGB 대신 전체 채널에 적용)
def random_brightness(image, max_delta=0.1, seed=None):
    return tf.image.random_brightness(image, max_delta=max_delta, seed=seed)

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


    return image

def random_data_augmentation(image, seed=None):
    """
    image: (12, 12, 20) 크기의 입력 이미지
    crop_size: 크롭 후 목표 크기
    """
    # 입력 이미지가 0~1 사이 값이라고 가정
    # image = tf.ensure_shape(image, [12, 12, 20])
    # crop_size = image.shape
    # 1. 랜덤 크롭
    image = random_crop(image, seed=seed)
    
    prob_ = []
    for _ in range(4):
        prob = np.random.uniform(0,1, [1])
        prob_.append(prob)

    if prob_[0] > 0.5:
        image = random_flip(image, seed=seed)
    
    if prob_[1] > 0.5:
        image = random_rotation(image, seed=seed)
    
    if prob_[2] > 0.5:
        image = random_brightness(image, max_delta=0.1, seed=seed)
    
    if prob_[3] > 0.5:
        image = gaussian_noise(image)


    return image
# 예제 사용
def apply_simclr_augmentation(image, aug_1, aug_2, random = False):
    # 두 개의 서로 다른 증강된 뷰 생성 (SimCLR의 핵심)
    #seed1 = tf.random.uniform([1], maxval=1000, dtype=tf.int32)
    #seed2 = tf.random.uniform([1], maxval=1000, dtype=tf.int32)
    
    if random:   
        view1 = random_data_augmentation(image)
        view2 = random_data_augmentation(image)
    else:
        view1 = data_augmentation(image, aug_1)
        view2 = data_augmentation(image, aug_2)    


    return view1, view2
# 예제 사용