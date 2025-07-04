import numpy as np
from data.patches import *

def patch_(height, width, tif_img, patch_size, stride):
    """patch_size : half of the patch size"""
    coor = get_coordinate(height, width)
    coor_patch_list = get_patches_coors(coor, patch_size, stride)
    data_list = get_patches_data(tif_img, patch_size, stride)

    ls_list = data_list[20]

    coor_width = coor_patch_list[0]
    coor_width = np.expand_dims(coor_width, axis=3)

    coor_height = coor_patch_list[1]
    coor_height = np.expand_dims(coor_height, axis=3)

    coor_ = np.concatenate((coor_width, coor_height), axis=3)

    data_ = data_list[0]
    data_ = np.expand_dims(data_, axis=3)

    for i in range(19):
        data_num = data_list[i+1]
        data_num = np.expand_dims(data_num, axis=3)
        data_ = np.concatenate([data_, data_num], axis=3)

    real_patches = []
    real_ls = []
    coor_list = []
    for i in range(ls_list.shape[0]):
        if len(np.unique(np.isnan(data_[i,:,:,0]), return_counts=True)[0]) ==2: # 셀 안에 Nan 값이 포함될 경우
            if np.sum(np.isnan(data_[i,patch_size-1:patch_size+1,patch_size-1:patch_size+1,:]))==0: # 중앙 네 개의 셀 중에 하나라도 Nan이 있어서는 안됨
                real_patch = data_[i:i+1]
                real_ls_patch = ls_list[i:i+1]
                coor_list.append(coor_[i:i+1])
                real_patches.append(real_patch)
                real_ls.append(real_ls_patch)
        else:
            if np.unique(np.isnan(data_[i,:,:,0]), return_counts=True)[0] ==False:
                if np.sum(np.isnan(data_[i,patch_size-1:patch_size+1,patch_size-1:patch_size+1,:]))==0: # 중앙 네 개의 셀 중에 하나라도 Nan이 있어서는 안됨
                    real_patch = data_[i:i+1]
                    real_ls_patch = ls_list[i:i+1]
                    coor_list.append(coor_[i:i+1])
                    real_patches.append(real_patch)
                    real_ls.append(real_ls_patch)

    coor_list = np.concatenate(coor_list)
    real_patches = np.concatenate(real_patches)
    real_ls = np.concatenate(real_ls)
    real_ls = np.expand_dims(real_ls, axis=3)

    #real_patches[:,:,:,4] = np.log10(real_patches[:,:,:,4])
    #real_patches[:,:,:,11] = np.log10(real_patches[:,:,:,11])
    #real_patches[:,:,:,12] = np.sqrt(real_patches[:,:,:,12])
    #real_patches[:,:,:,-1] = np.log10(real_patches[:,:,:,-1])

    # 0 -> 0.001, nan -> 0
    real_ls[np.where(np.isnan(real_patches[:,:,:,0:1]))] = -0.001

    real_ls = real_ls + 0.001
    real_ls[real_ls==1.001] = 1

    ls_where = np.where(real_ls[:, 5:7, 5:7, :]==1.) 

    numbers = np.arange(0, len(real_patches), 1) 
    numbers = np.delete(numbers, np.unique(ls_where[0], return_counts=True)[0]) 

    coor_non_ls = coor_list[numbers]
    coor_ls = coor_list[np.unique(ls_where[0], return_counts=True)[0]]

    non_ls_patches_features = real_patches[numbers]
    ls_patches_features = real_patches[np.unique(ls_where[0], return_counts=True)[0]]

    non_ls_patches_labels = real_ls[numbers]
    ls_patches_labels = real_ls[np.unique(ls_where[0], return_counts=True)[0]]

    return real_patches, non_ls_patches_features, ls_patches_features, non_ls_patches_labels, ls_patches_labels, coor_list
