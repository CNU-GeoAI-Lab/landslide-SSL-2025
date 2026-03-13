import numpy as np

def get_coordinate(height, width):
    """
    height : list or array of (height min, height max, number of pixel)
    width : list or array of (width min, width max, number of pixel)
    """
    height_coor = np.linspace(height[1], height[0], height[2])
    width_coor = np.linspace(width[0], width[1], width[2])

    height_coor, width_coor = np.meshgrid(height_coor, width_coor, indexing='ij')

    height_coor = np.expand_dims(height_coor, axis=0)
    width_coor = np.expand_dims(width_coor, axis=0)

    full_coor_ = np.concatenate((width_coor, height_coor), axis=0)

    return full_coor_

def get_patches_data(image, patch_size_half, stride):
    data_list = []
    for a in range(21):
        patch_list = []
        tif_test = image[a]
        for i in range(patch_size_half, tif_test.shape[0]-patch_size_half, stride):
            for j in range(patch_size_half, tif_test.shape[1]-patch_size_half, stride):
                patch = tif_test[i-patch_size_half:i+patch_size_half, j-patch_size_half:j+patch_size_half]
                patch = np.expand_dims(patch, axis=0)
                patch_list.append(patch)
        data = np.concatenate(patch_list)
        data = np.expand_dims(data, axis=0)
        data_list.append(data)

    data_list = np.concatenate(data_list)
    
    return data_list

def get_patches_coors(coor, patch_size_half, stride):
    coor_patch_list = []
    for a in range(2):
        coor_patch_list_ = []
        coor_num = coor[a]
        for i in range(patch_size_half, coor_num.shape[0]-patch_size_half, stride):
            for j in range(patch_size_half, coor_num.shape[1]-patch_size_half, stride):
                patch = coor_num[i-patch_size_half:i+patch_size_half, j-patch_size_half:j+patch_size_half]
                patch = np.expand_dims(patch, axis=0)
                coor_patch_list_.append(patch)
        coor_ = np.concatenate(coor_patch_list_)
        coor_ = np.expand_dims(coor_, axis=0)
        coor_patch_list.append(coor_)

    coor_patch_list = np.concatenate(coor_patch_list)
    
    return coor_patch_list
