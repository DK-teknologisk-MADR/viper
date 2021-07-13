from toy_file import get_file_pairs
import numpy as np
import shutil
import os
data_dir_from = "/home/madr/Projects/RK/Imitation_learning/Viper_data/viper_data_june/all_data_backup/data"
data_dir = "/home/madr/Projects/RK/Imitation_learning/Viper_data/viper_data_june"
files =  get_file_pairs(data_dir_from,"")
files
tup = len(files)*.8,len(files)*.1,len(files)*.1
ls = [int(x) for x in tup]
split_types = ["train","val","test"]
splits = np.repeat(split_types,ls)
np.random.shuffle(splits)
for split in split_types:
    os.makedirs(os.path.join(data_dir,split),exist_ok=True)
for fr_file_pair,split in zip(files.items(),splits):
    front,file_ls = fr_file_pair
    for file in file_ls:
        shutil.move(os.path.join(data_dir_from, "", file ), os.path.join(data_dir, split, file ) )
