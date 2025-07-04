import os
import numpy as np
import argparse
from tensorflow.python.client import device_lib
from data.data_reader import *
from data.augmentation import *
from loss.loss import *
from train.train import *
from utils.utils import *
from model.simclr_model import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, cohen_kappa_score

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

from tensorflow.keras.applications import ResNet50

def main():
    parser = argparse.ArgumentParser(description="Train a SimCLR on Landslide dataset")
    parser.add_argument("--patch_size", type = int, default = 6, help="half size of patch")
    parser.add_argument("--pre_batch_size", type = int, default = 128, help = "batch size of the pretraining model")
    parser.add_argument("--batch_size", type = int, default = 32, help = "batch size of fine tuning model")
    parser.add_argument("--pre_epochs", type = int, default = 1, help = "epochs of pre training model")
    parser.add_argument("--strides", type = int, default = 6, help = "strides when make patch from image")

    parser.add_argument("--pre_learning_rate", type = float, default = 0.0001, help = "learning rate of pretraining model")
    parser.add_argument("--learning_rate", type = float, default = 0.00001, help = "learning rate of fine tuning model")
    parser.add_argument("--fine_tuning_data_ratio", type = float, default = 1e-1, help = "ratio of dataset used on fine tuing")

    parser.add_argument("--height", type = list, default = [554270.0, 562860.0, 859], help = "information of height")
    parser.add_argument("--width", type = list, default = [331270.0, 342070.0, 1080], help = "information of width")   

    parser.add_argument("--aug_1", nargs = "+", type = list, default = [1, 1, 1, 1], help = "information of first augmentation of SimCLR")
    parser.add_argument("--aug_2", nargs = "+", type = list, default = [1, 1, 1, 1], help = "information of second augmentation of SimCLR")

    parser.add_argument("--flip", type = bool, default = True, help = "Whether to have flip augmentation")
    parser.add_argument("--rotation", type = bool, default = True, help = "Whether to have rotation augmentation")
    parser.add_argument("--brightness", type = bool, default = True, help = "Whether to have brightness augmentation")
    parser.add_argument("--gaussian", type = bool, default = True, help = "Whether to have gaussian augmentation")
    parser.add_argument("--random_aug", type = bool, default = False, help = "Whether to apply random augmentation or not")

    parser.add_argument("--tif_img_path", type = str, default = './dataset/tif_img.npy', help = "tif image path")
    parser.add_argument("--ls_img_path", type = str, default = './dataset/ls_img.npy', help = "landslide image path")
    parser.add_argument("--pre_trained_model_name", type = str, default = "pre", help = "name of the pre-trained model")
    parser.add_argument("--fine_trained_model_name", type = str, default = "fine", help = "name of the fine-trained model")
    parser.add_argument("--dir_name", type = str, default = "250228", help = "directory name of fine-tuned model to save")
    parser.add_argument("--pre_model", type = str, default = "ResNet", help = "['ResNet', 'CNN', 'ViT'] sort of pre-training model")
    parser.add_argument("--ssl_type", type = str, default = "SimSiam", help = "['SimCLR', 'BYOL', 'MoCo', 'SimSiam'] sort of self-supervised learning model")

    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"


    tif_img = np.load(args.tif_img_path)
    ls_img = np.load(args.ls_img_path)
    ls_img = np.expand_dims(ls_img, 0)
    tif_img = np.concatenate([tif_img, ls_img], axis=0)

    real_patches, non_ls_patches_features, ls_patches_features, non_ls_patches_labels, ls_patches_labels, coor_list = patch_(args.height, args.width, tif_img, args.patch_size, args.strides)

    #real_patches = real_patches[:,2:-2,2:-2,:]

    multi_data_scale = Multi_data_scaler(real_patches)
    train_scaled = multi_data_scale.multi_scale(real_patches)
    
    train_scaled = tf.image.resize(train_scaled, (28, 28))
    train_scaled = np.array(train_scaled)

    np.random.seed(114)
    np.random.shuffle(train_scaled)

    data_1 = []
    data_2 = []
    for i in range(len(train_scaled)):
        view1, view2 = apply_simclr_augmentation(train_scaled[i], args.aug_1, args.aug_2, args.random_aug)
        view1 = np.expand_dims(view1, axis=0)
        view2 = np.expand_dims(view2, axis=0)
        data_1.append(view1)
        data_2.append(view2)

    data_1 = np.concatenate(data_1)
    data_2 = np.concatenate(data_2)
    
    print(data_1.shape)
    print(data_2.shape)

    try:
        os.makedir('./pretrained_model/%s'%args.dir_name)
    except:
        pass

    if args.ssl_type == "BYOL":
        with tf.device('/GPU:0'):
            pretrain_model = pre_train_byol(data_1, data_2, batch_size=args.pre_batch_size, pre_model=args.pre_model, epochs=args.pre_epochs)          
    elif args.ssl_type == "SimCLR":
        with tf.device('/GPU:0'):
            pretrain_model = pre_train_simclr(data_1, data_2, batch_size=args.pre_batch_size, pre_model=args.pre_model, epochs=args.pre_epochs)
    elif args.ssl_type == "MoCo":
        with tf.device('/GPU:0'):
            pretrain_model = pre_train_moco(data_1, data_2, batch_size=args.pre_batch_size, pre_model=args.pre_model, epochs=args.pre_epochs)
    elif args.ssl_type == "SimSiam":
        with tf.device('/GPU:0'):
            pretrain_model = pre_train_simsiam(data_1, data_2, batch_size=args.pre_batch_size, pre_model=args.pre_model, epochs=args.pre_epochs)

    try:
        pretrain_model.save_weights("./pretrained_model/%s/%s.h5"%(args.dir_name,args.pre_trained_model_name))
    except:
        pass

    ls_patches_features = np.load("./dataset/train/ls_patches.npy")
    non_ls_patches_features = np.load("./dataset/train/non_ls_patches.npy")
    ls_patches_labels = np.load("./dataset/train/ls.npy")
    non_ls_patches_labels = np.load("./dataset/train/non_ls.npy")
    
    ls_test_patches = np.load("./dataset/test/ls_patches.npy")
    non_ls_test_patches = np.load("./dataset/test/non_ls_patches.npy")
    ls_test = np.load("./dataset/test/ls.npy")
    non_ls_test = np.load("./dataset/test/non_ls.npy")

    #ls_patches_features = ls_patches_features[:,2:-2,2:-2,:]
    #non_ls_patches_features = non_ls_patches_features[:,2:-2,2:-2,:]
    #ls_patches_labels = ls_patches_labels[:,2:-2,2:-2,:]
    #non_ls_patches_labels = non_ls_patches_labels[:,2:-2,2:-2,:]

    #ls_test_patches = ls_test_patches[:,2:-2,2:-2,:]
    #non_ls_test_patches = non_ls_test_patches[:,2:-2,2:-2,:]
    #ls_test = ls_test[:,2:-2,2:-2,:]
    #non_ls_test = non_ls_test[:,2:-2,2:-2,:]
    
    valid_non_ls_ind = np.random.choice(non_ls_patches_features.shape[0], int(ls_patches_labels.shape[0]*0.2))
    valid_ls_ind = np.random.choice(ls_patches_features.shape[0], int(ls_patches_features.shape[0]*0.2))

    valid_non_ls = non_ls_patches_features[valid_non_ls_ind]
    valid_non_ls_label = non_ls_patches_labels[valid_non_ls_ind]
    valid_ls = ls_patches_features[valid_ls_ind]
    valid_ls_label = ls_patches_labels[valid_ls_ind]
    
    non_ls_patches_features = np.delete(non_ls_patches_features, valid_non_ls_ind, axis=0)
    non_ls_patches_labels = np.delete(non_ls_patches_labels, valid_non_ls_ind, axis=0)
    ls_patches_features = np.delete(ls_patches_features, valid_ls_ind, axis=0)
    ls_patches_labels = np.delete(ls_patches_labels, valid_ls_ind, axis=0)
    
#    valid_non_ls = valid_non_ls[:int(len(valid_non_ls_ind)*args.fine_tuning_data_ratio)]
#    valid_non_ls_label = valid_non_ls_label[:int(len(valid_non_ls_ind)*args.fine_tuning_data_ratio)]
#    valid_ls = valid_ls[:int(len(valid_ls_ind)*args.fine_tuning_data_ratio)]
#    valid_ls_label = valid_ls_label[:int(len(valid_ls_ind)*args.fine_tuning_data_ratio)]

    train_non_ls_ind = np.random.choice(non_ls_patches_features.shape[0], int(ls_patches_labels.shape[0]*2))
    train_ls_ind = np.random.choice(ls_patches_features.shape[0], int(ls_patches_features.shape[0]))

    train_non_ls_ind = train_non_ls_ind[:int(len(train_non_ls_ind)*args.fine_tuning_data_ratio)]
    train_ls_ind = train_ls_ind[:int(len(train_ls_ind)*args.fine_tuning_data_ratio)]

    train_non_ls = non_ls_patches_features[train_non_ls_ind]
    train_non_ls_label = non_ls_patches_labels[train_non_ls_ind]
    train_ls = ls_patches_features[train_ls_ind]
    train_ls_label = ls_patches_labels[train_ls_ind]    

    train_patches = np.concatenate([train_ls, train_non_ls])
    valid_patches = np.concatenate([valid_ls, valid_non_ls])
    train_labels = np.concatenate([train_ls_label, train_non_ls_label])
    valid_labels = np.concatenate([valid_ls_label, valid_non_ls_label])

    train_labels_ = []
    valid_labels_ = []
    for i in range(train_labels.shape[0]):
        if np.sum(train_labels[i, args.patch_size-1:args.patch_size+1, args.patch_size-1:args.patch_size+1]) >=1.:
            ls_bool = 1
            train_labels_.append(ls_bool)
        else:
            ls_bool = 0
            train_labels_.append(ls_bool)

    for i in range(valid_labels.shape[0]):
        if np.sum(valid_labels[i, args.patch_size-1:args.patch_size+1, args.patch_size-1:args.patch_size+1]) >=1.:
            ls_bool = 1
            valid_labels_.append(ls_bool)
        else:
            ls_bool = 0
            valid_labels_.append(ls_bool)
    
    _patch_size = 28

    train_labels_ = np.expand_dims(train_labels_, axis=1)
    valid_labels_ = np.expand_dims(valid_labels_, axis=1)
    
    train_patches = tf.image.resize(train_patches, (_patch_size, _patch_size))
    valid_patches = tf.image.resize(valid_patches, (_patch_size, _patch_size))
    
    train_patches = np.array(train_patches)
    valid_patches = np.array(valid_patches)

    train_patches = train_patches[:,5:-5, 5:-5, :]
    valid_patches = valid_patches[:,5:-5, 5:-5, :]

    train_patches_scaled = multi_data_scale.multi_scale(train_patches)
    valid_patches_scaled = multi_data_scale.multi_scale(valid_patches)

    input_shape = train_patches_scaled.shape[1:]

    if args.ssl_type == "SimCLR":
        encoder = pretrain_model.get_layer("encoder")
    elif args.ssl_type == "BYOL":
        encoder = pretrain_model.online_encoder.encoder
    elif args.ssl_type == "MoCo":
        encoder = pretrain_model.encoder_q
    elif args.ssl_type == "SimSiam":
        encoder = pretrain_model.encoder.encoder

    finetune_model = build_finetune_model(input_shape, encoder, num_classes=2, training=False)
    
    finetune_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                        metrics=['accuracy'])
    try:
        os.makedirs('./finetuned_models/%s'%args.dir_name)
    except:
        print("Directory name %s"%args.dir_name, 'exist')

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("./finetuned_models/%s/%s.h5"%(args.dir_name, args.fine_trained_model_name), save_best_only=True)
    checkpoint_cb2 = tf.keras.callbacks.ModelCheckpoint("./finetuned_models/%s/%s_weight.h5"%(args.dir_name, args.fine_trained_model_name), save_weight_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=20,
                                                    restore_best_weights=True)
    reduce_lr_plateu_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)

    with tf.device('/GPU:0'):
        history_2 = finetune_model.fit(train_patches_scaled, train_labels_,
                        validation_data = (valid_patches_scaled, valid_labels_),
                        epochs=10,
                        batch_size=32,
                        callbacks = [checkpoint_cb,
                                checkpoint_cb2,
                                early_stopping_cb,
                                reduce_lr_plateu_cb,
                                    ]
                                    )
    try:
        os.makedirs('./loss_curve/%s'%args.dir_name)
    except:
        pass
    plt.plot(history_2.history['val_loss'], label = 'valid_loss')
    plt.plot(history_2.history['loss'], label = 'train_loss')
    plt.legend()
    plt.grid(True)
    plt.xlabel('epochs')
    plt.ylabel('Categorical_Cross_Entropy')
    plt.title('Loss curve')
    plt.ylim(0.001,1.)
    plt.tight_layout()
    plt.savefig('./loss_curve/%s/%s'%(args.dir_name,args.fine_trained_model_name))

    ls_patches_features = np.load("./dataset/train/ls_patches.npy")
    non_ls_patches_features = np.load("./dataset/train/non_ls_patches.npy")
    ls_patches_labels = np.load("./dataset/train/ls.npy")
    non_ls_patches_labels = np.load("./dataset/train/non_ls.npy")

    ls_test_patches = np.load("./dataset/test/ls_patches.npy")
    non_ls_test_patches = np.load("./dataset/test/non_ls_patches.npy")
    ls_test = np.load("./dataset/test/ls.npy")
    non_ls_test = np.load("./dataset/test/non_ls.npy")

    ls_patches_features = multi_data_scale.multi_scale(ls_patches_features)
    non_ls_patches_features = multi_data_scale.multi_scale(non_ls_patches_features)   
    ls_test_patches = multi_data_scale.multi_scale(ls_test_patches)
    non_ls_test_patches = multi_data_scale.multi_scale(non_ls_test_patches)

    with tf.device('/CPU:0'):
        ls_patches_features = tf.image.resize(ls_patches_features, (_patch_size, _patch_size))  # shape = 6400, 18, 18, 20
        ls_patches_features = np.array(ls_patches_features)
        ls_patches_features = ls_patches_features[:,5:-5,5:-5,:]
        
        non_ls_patches_features = tf.image.resize(non_ls_patches_features, (_patch_size, _patch_size))  # shape = 6400, 18, 18, 20
        non_ls_patches_features = np.array(non_ls_patches_features)
        print(non_ls_patches_features.shape)
        non_ls_patches_features = non_ls_patches_features[:,5:-5,5:-5,:]
        
        ls_test_patches = tf.image.resize(ls_test_patches, (_patch_size, _patch_size))  # shape = 6400, 18, 18, 20
        ls_test_patches = np.array(ls_test_patches)
        ls_test_patches = ls_test_patches[:,5:-5,5:-5,:]
        
        non_ls_test_patches = tf.image.resize(non_ls_test_patches, (_patch_size, _patch_size))  # shape = 6400, 18, 18, 20
        non_ls_test_patches = np.array(non_ls_test_patches)
        non_ls_test_patches = non_ls_test_patches[:,5:-5,5:-5,:]
    
    full_features = np.concatenate((ls_patches_features, non_ls_patches_features, ls_test_patches, non_ls_test_patches), axis=0)
    
    with tf.device('/GPU:0'):
        full_result = finetune_model.predict(full_features)


    train_ls_coor = np.load('./dataset/train/ls_coor.npy')
    train_non_ls_coor = np.load('./dataset/train/non_ls_coor.npy')
    test_ls_coor = np.load('./dataset/test/ls_coor.npy')
    test_ls_non_coor = np.load('./dataset/test/non_ls_coor.npy')

    coor_list = np.concatenate((train_ls_coor, train_non_ls_coor, test_ls_coor, test_ls_non_coor),axis=0)

    plt.rc('font', size=15) 

    cmap = ListedColormap(['darkgreen', 'lawngreen', 'palegoldenrod','coral','firebrick'])
    variable_list = []
    caution = ['Very Low','Low','Moderate','High','Very High']
    for i in range(5):
        variable = mpatches.Patch(color = cmap.colors[i], label='%s'%caution[i])
        variable_list.append(variable)

    variable_list.append(Line2D([0], [0], marker='o', color='w', label = 'Landslide', markerfacecolor='k', markersize=3))

    ls_coor = np.load('./dataset/ls_coor.npy')
    ls_label = np.load('./dataset/ls_labels.npy')

    try:
        os.makedirs('./images/landslide_maps/%s'%args.dir_name)
    except:
        pass        

    plt.figure(figsize=(10, 6))
    plt.scatter(coor_list[:,5,5,0], coor_list[:,5,5,1], s= 0.1, c = full_result[:,1], cmap= "RdYlGn_r")
    plt.scatter(ls_coor[:,0], ls_coor[:,1], s=3, c=ls_label[:,0], cmap='gray')
    plt.legend(handles =variable_list, bbox_to_anchor=(1.02, 1))

    plt.title('%s susceptibility'%args.fine_trained_model_name)
    plt.xlabel('Easting')
    plt.ylabel('Northing')
    plt.tight_layout()
    plt.savefig('./images/landslide_maps/%s/%s.png'%(args.dir_name,args.fine_trained_model_name))

if __name__ == "__main__":
    main()