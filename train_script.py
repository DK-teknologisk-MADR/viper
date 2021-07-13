import shutil

import toy_file as ty
import cv2
import os
import pandas as pd
import numpy as np

split = 'train'
base_dir = '/home/madr/Projects/RK/Imitation_learning/Viper_data'
data_dir = os.path.join(base_dir, 'viper_data_june')
#cols = [" camX", "camY"," camZ"]
#norm_dims = [0, 1,2]
cols = [" camX", "camY"]
norm_dims = [0, 1]

crop_dims = 180,590,420,830
dim = len(norm_dims)
resize_nr = 75
n = 50
n_kpts = 12

pairs = ty.get_file_pairs(data_dir,split)
img = cv2.imread(os.path.join(data_dir,split,pairs[1][0]))
img = img[crop_dims[0] : crop_dims[1], crop_dims[2]:crop_dims[3]]
img = cv2.resize(img,(resize_nr,resize_nr))

ty.load_treat_and_jsonize(base_dir,data_dir,cols,norm_dims,n,nkey=n_kpts,resize_dim = (resize_nr,resize_nr),crop_dims=crop_dims)
shell_str = ['python','train.py',
             f"--train_dir=/home/madr/Projects/RK/Imitation_learning/Viper_data/fits_nc{n_kpts}_steps{n}_size{resize_nr}_dim{dim}/annotations/train",
             f"--valid_dir=/home/madr/Projects/RK/Imitation_learning/Viper_data/fits_nc{n_kpts}_steps{n}_size{resize_nr}_dim{dim}/annotations/val",
             "--model_name=Model_RK2021_lego_coefs",
             "--num_epochs=2000",
             "--label_type=RAW",
             "--batch_size=4"]
print(" ".join(shell_str))


