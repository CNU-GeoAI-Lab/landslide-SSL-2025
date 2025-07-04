#!/bin/bash
# source ~/anaconda3/bin/activate transform

# python main.py --random_aug True --ssl_type 'SimSiam' --pre_trained_model_name 'simsiam_1' --fine_trained_model_name 'simsiam_1' --fine_tuning_data_ratio 1e-2 --dir_name 'simsiam_vit_0704' --model_type 'ViT' --pre_epochs 25 --strides 3
# python main.py --random_aug True --ssl_type 'SimSiam' --pre_trained_model_name 'simsiam_5' --fine_trained_model_name 'simsiam_5' --fine_tuning_data_ratio 5e-2 --dir_name 'simsiam_vit_0704' --model_type 'ViT' --pre_epochs 25 --strides 3
# python main.py --random_aug True --ssl_type 'SimSiam' --pre_trained_model_name 'simsiam_10' --fine_trained_model_name 'simsiam_10' --fine_tuning_data_ratio 1e-1 --dir_name 'simsiam_vit_0704' --model_type 'ViT' --pre_epochs 25 --strides 3
# python main.py --random_aug True --ssl_type 'SimSiam' --pre_trained_model_name 'simsiam_20' --fine_trained_model_name 'simsiam_20' --fine_tuning_data_ratio 2e-1 --dir_name 'simsiam_vit_0704' --model_type 'ViT' --pre_epochs 25 --strides 3
# python main.py --random_aug True --ssl_type 'SimSiam' --pre_trained_model_name 'simsiam_30' --fine_trained_model_name 'simsiam_30' --fine_tuning_data_ratio 3e-1 --dir_name 'simsiam_vit_0704' --model_type 'ViT' --pre_epochs 25 --strides 3
# python main.py --random_aug True --ssl_type 'SimSiam' --pre_trained_model_name 'simsiam_40' --fine_trained_model_name 'simsiam_40' --fine_tuning_data_ratio 4e-1 --dir_name 'simsiam_vit_0704' --model_type 'ViT' --pre_epochs 25 --strides 3
# python main.py --random_aug True --ssl_type 'SimSiam' --pre_trained_model_name 'simsiam_50' --fine_trained_model_name 'simsiam_50' --fine_tuning_data_ratio 5e-1 --dir_name 'simsiam_vit_0704' --model_type 'ViT' --pre_epochs 25 --strides 3
# python main.py --random_aug True --ssl_type 'SimSiam' --pre_trained_model_name 'simsiam_60' --fine_trained_model_name 'simsiam_60' --fine_tuning_data_ratio 6e-1 --dir_name 'simsiam_vit_0704' --model_type 'ViT' --pre_epochs 25 --strides 3
# python main.py --random_aug True --ssl_type 'SimSiam' --pre_trained_model_name 'simsiam_70' --fine_trained_model_name 'simsiam_70' --fine_tuning_data_ratio 7e-1 --dir_name 'simsiam_vit_0704' --model_type 'ViT' --pre_epochs 25 --strides 3
# python main.py --random_aug True --ssl_type 'SimSiam' --pre_trained_model_name 'simsiam_80' --fine_trained_model_name 'simsiam_80' --fine_tuning_data_ratio 8e-1 --dir_name 'simsiam_vit_0704' --model_type 'ViT' --pre_epochs 25 --strides 3
# python main.py --random_aug True --ssl_type 'SimSiam' --pre_trained_model_name 'simsiam_90' --fine_trained_model_name 'simsiam_90' --fine_tuning_data_ratio 9e-1 --dir_name 'simsiam_vit_0704' --model_type 'ViT' --pre_epochs 25 --strides 3
# python main.py --random_aug True --ssl_type 'SimSiam' --pre_trained_model_name 'simsiam_full' --fine_trained_model_name 'simsiam_full' --fine_tuning_data_ratio 1 --dir_name 'simsiam_vit_0704' --model_type 'ViT' --pre_epochs 25 --strides 3

# python main.py --ssl_type 'no_pretrain' --fine_trained_model_name 'vit_sup_1' --fine_tuning_data_ratio 1e-2 --dir_name 'vit_sup_0703' --model_type 'ViT' --strides 3
# python main.py --ssl_type 'no_pretrain' --fine_trained_model_name 'vit_sup_5' --fine_tuning_data_ratio 5e-2 --dir_name 'vit_sup_0703' --model_type 'ViT' --strides 3
# python main.py --ssl_type 'no_pretrain' --fine_trained_model_name 'vit_sup_10' --fine_tuning_data_ratio 1e-1 --dir_name 'vit_sup_0703' --model_type 'ViT' --strides 3
# python main.py --ssl_type 'no_pretrain' --fine_trained_model_name 'vit_sup_20' --fine_tuning_data_ratio 2e-1 --dir_name 'vit_sup_0703' --model_type 'ViT' --strides 3
# python main.py --ssl_type 'no_pretrain' --fine_trained_model_name 'vit_sup_30' --fine_tuning_data_ratio 3e-1 --dir_name 'vit_sup_0703' --model_type 'ViT' --strides 3
# python main.py --ssl_type 'no_pretrain' --fine_trained_model_name 'vit_sup_40' --fine_tuning_data_ratio 4e-1 --dir_name 'vit_sup_0703' --model_type 'ViT' --strides 3
# python main.py --ssl_type 'no_pretrain' --fine_trained_model_name 'vit_sup_50' --fine_tuning_data_ratio 5e-1 --dir_name 'vit_sup_0703' --model_type 'ViT' --strides 3
# python main.py --ssl_type 'no_pretrain' --fine_trained_model_name 'vit_sup_60' --fine_tuning_data_ratio 6e-1 --dir_name 'vit_sup_0703' --model_type 'ViT' --strides 3
# python main.py --ssl_type 'no_pretrain' --fine_trained_model_name 'vit_sup_70' --fine_tuning_data_ratio 7e-1 --dir_name 'vit_sup_0703' --model_type 'ViT' --strides 3
# python main.py --ssl_type 'no_pretrain' --fine_trained_model_name 'vit_sup_80' --fine_tuning_data_ratio 8e-1 --dir_name 'vit_sup_0703' --model_type 'ViT' --strides 3
# python main.py --ssl_type 'no_pretrain' --fine_trained_model_name 'vit_sup_90' --fine_tuning_data_ratio 9e-1 --dir_name 'vit_sup_0703' --model_type 'ViT' --strides 3
# python main.py --ssl_type 'no_pretrain' --fine_trained_model_name 'vit_sup_full' --fine_tuning_data_ratio 1 --dir_name 'vit_sup_0703' --model_type 'ViT' --strides 3
