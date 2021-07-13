from trajectory_estimator import TrajectoryEstimator
import torch
import os
import json
from numba import cuda
import pandas as pd
import numpy as np
import pandas as pd
import re
import cv2
import time
import subprocess
import validater
import sys
sys.path.append(os.path.join(os.getcwd(), 'C073_Robotcell_keypoints/tf_model1/tf_model1'))
from model_tester import ModelTester
pd.set_option('display.max_columns', 500)
pd.set_option("display.precision", 3)
from shutil import copy
import matplotlib.pyplot as plt
#traj_est = TrajectoryEstimator()



def dump_json(output_path, image_path, image_height, image_width, points, label, shape_type='polygon'):
    assert type(points) == np.ndarray
    data = dict()
    data['imagePath'] = image_path
    data['imageData'] = None
    shape_entry = dict()
    shape_entry['label'] = label
    shape_entry['shape_type'] = shape_type
    shape_entry['points'] = points.tolist()
    data['shapes'] = [shape_entry]
    data['imageWidth'] = image_width
    data['imageHeight'] = image_height
    with open(output_path, "w") as fout:
        json.dump(data, fout, indent=2)



def get_file_pairs(data_dir,split):
    file_dict = {}
    reg = re.compile(r"_([0-9][0-9]?[0-9]?)((_vp.txt)|(_rgb.jpg))")
    file_dir = os.path.join(data_dir, split)
    files = os.listdir(file_dir)
    for file in files:
        ma = reg.search(file)
        if ma:
            nr = int ( ma.group(1))
            if nr not in file_dict:
                file_dict[nr] = []
            file_dict[nr].append(file)
            file_dict[nr].sort(key = lambda str : str.split(".")[-1])
    return file_dict

def resample(arr, n,equi_distance_dims = None):
    '''
        # arr = [[[x_00, y_00, z_00, ox_00, oy_00, oz_00],
        #          [x_01, y_01, z_01, ox_01, oy_01, oz_01],
        #          [...],
        #          [x_0k, y_0k, z_0k, ox_0k, oy_0k, oz_0k]],
        #         [[...],
        #          [...],
        #          [...]],
        #         [[x_N0, y_N0, z_N0, ox_N0, oy_N0, oz_N0],
        #          [x_N1, y_N1, z_N1, ox_N1, oy_N1, oz_n1],
        #          ...
        #          [x_Nk, y_Nk, z_Nk, ox_Nk, oy_Nk, oz_Nk]]
        #
    Resamples points and makes them equidistant in a subspace of R^d, where d=arr.shape[0].
    equi_distance_dims is the list of dimensions which gives the subspace that equi-distance is calculated of. If none, the entire space R^d is taken.
    '''
    if equi_distance_dims is None:
        equi_distance_dims = [ i for i in range(arr.shape[1])]
    # interpolate between data points
    #we treat the array as a curve with "fake" time steps given by uniform distance up to t=10000. Then we interpolate x(t),y(t),z(t) etc...
    time= 10000
    obs_nr = arr.shape[0]
    time_steps_fake = np.floor(np.linspace(0, time, obs_nr))
    time_pts = np.array(range(time))
    interp_curve = np.apply_along_axis(func1d=lambda ar: np.interp(time_pts, time_steps_fake, ar), arr=arr, axis=0)
    # resample to n points (equidistant along path)
    df = np.diff(interp_curve,axis=0)
    dnorms = np.linalg.norm(df[:,equi_distance_dims],axis=1)
    u = np.cumsum(dnorms,axis=0)
    u = np.hstack([[0], u])
    t_result = np.linspace(0, u.max(), n)
    result = np.apply_along_axis(func1d=lambda ar: np.interp(t_result, u, ar), arr=interp_curve, axis=0)
    return result

def load_data(data_dir,split):
    dfs = {}
    file_pairs = get_file_pairs(data_dir,split)
    for i,files in enumerate(file_pairs.items()):
        key, ls = files
        print(files)
        df = pd.read_csv(os.path.join(data_dir,split, ls[1] ) )
        nan_mask = df.isna()
        if np.any(nan_mask.any(axis=1)):
            df = df.dropna()
            print(f'dropped {nan_mask.any(axis=1).sum()} rows due to NaNs in data indexed {i} named {ls[1]}')
        dfs[key] = df
    return dfs

def dfs_to_svd_input(dfs,split,cols,dims_to_normalize,n=60):
    #    if all(isinstance(col, str) for col in cols):
    #        pass
    #    elif all(isinstance(col, int) for col in cols):
    #        pass
    #    else:
    #        raise ValueError("cols should be a list of either ints or strings")
    id_from_index = []
    dims_n = len(cols)
    obs_n = len(dfs)
    svd_input = np.zeros((obs_n,n,dims_n))
    for ind,df_pair in enumerate(dfs.items()):
        ID,df = df_pair
        print(df.columns)
        curve_dt = df[cols].values
        id_from_index.append(ID)
        svd_input[ind,:] = resample(curve_dt, n, dims_to_normalize)
    return svd_input,id_from_index


def get_out_dir_suffix(nc,steps,dim,resize_dim):
    return f'fits_nc{nc}_steps{steps}_size{resize_dim}_dim{dim}'

def data_transform(img,resize_dims,crop_dims):
    img = img[crop_dims[0]: crop_dims[1], crop_dims[2]:crop_dims[3]]
    img = cv2.resize(img, resize_dims)
    return img

def get_normalized_coefs(svd_pair,traj_est,predef_mean_std_tuple = None):
    '''
    returns dict of coefs, mean, svd, norm_coefs after (coefs-mean)/svd
    '''
    svd_input, id_from_index = svd_pair
    svd_input_reshaped = traj_est.create_reformed_data(svd_input)
    coefs = np.zeros((svd_input.shape[0],traj_est.nc))
    for idx in range(svd_input.shape[0]):
        target = svd_input_reshaped[idx,:]
        result = traj_est.fit_leastsq(target)
        coefs[idx] = result[0]
    if predef_mean_std_tuple is None:
        coef_mean = coefs.mean(axis=0)
        coef_std = coefs.std(axis=0)
    else:
        print(predef_mean_std_tuple)
        coef_mean,coef_std = predef_mean_std_tuple
    norm_coefs = (coefs-coef_mean) / coef_std
    result = ( coef_mean , coef_std, norm_coefs)
    return result

def prepare_and_dump_jsons(svd_pair,traj_est,data_dir,split,out_dir_base,resize_dim,crop_dims,predef_mean_std_tuple = None):
    assert isinstance(traj_est,TrajectoryEstimator)
    if out_dir_base is None:
        out_dir_base = os.getcwd()
    outdir = os.path.join(out_dir_base,
                          get_out_dir_suffix(traj_est.nc,svd_pair[0].shape[1],svd_pair[0].shape[2],resize_dim[0]),
                          'annotations',
                          split)
    print(f"copying to {outdir})")
    os.makedirs(outdir, exist_ok=True)
    files = get_file_pairs(data_dir,split)
    svd_input, id_from_index = svd_pair
    coef_mean,coef_std,norm_coefs = get_normalized_coefs(svd_pair = svd_pair, traj_est = traj_est, predef_mean_std_tuple=None)
    assert norm_coefs.shape[1] == traj_est.nc

# here we save parameters that are needed for transformation in testmode
    param_dict = {'norm_mean' : list(coef_mean),
                  'norm_std' : list(coef_std),
                  'resize_dim' : resize_dim,
                  'crop_dims' : crop_dims}
    param_dir = os.path.join(outdir,'transformation_data')
    os.makedirs(param_dir, exist_ok=True)
    with open(os.path.join(param_dir,'transformation_data.json'), "w+") as fout:
        json.dump(param_dict, fout, indent=2)

    #we now save each picture along with json data
    svd_input_reshaped = traj_est.create_reformed_data(svd_input)
    for idx in range(svd_input.shape[0]):
        id = id_from_index[idx]
        fname = files[id][0][:-4]
        img_name = files[id][0]
        img_path = os.path.join(data_dir,split,img_name)
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not find image {img_name}")
        image = data_transform(image,resize_dim,crop_dims)
        #target = svd_input_reshaped[idx,:]
        #result = traj_est.fit_leastsq(target)
        #est_target = traj_est.trajectory(result[0])
        dump_json(os.path.join(outdir, fname+'.json'), image_path=fname + '.jpg',
                  image_height=image.shape[0], image_width=image.shape[1], points=norm_coefs[idx], label='coefs',
                  shape_type='polygon')
        cv2.imwrite(os.path.join(outdir, fname+'.jpg'),image)


def traj_est_from_data_dir(data_dir,split,cols,norm_dims,n, nkey):
    dfs = load_data(data_dir, split)
    svd_train = dfs_to_svd_input(dfs, split, cols, norm_dims, n=n)
    traj_est = TrajectoryEstimator(svd_train[0])
    traj_est.calc_basisFcns(None, nkey)
    return traj_est


def load_treat_and_jsonize(base_dir,data_dir,cols,norm_dims,n, nkey,resize_dim,crop_dims):
    dfs = load_data(data_dir, 'train')
    svd_train = dfs_to_svd_input(dfs, 'train', cols, norm_dims, n=n)
    traj_est = TrajectoryEstimator(svd_train[0])
    traj_est.calc_basisFcns(None, nkey)
    dfs = load_data(data_dir, 'val')
    svd_val = dfs_to_svd_input(dfs, 'val', cols, norm_dims, n=n)
    dfs = load_data(data_dir, 'test')
    svd_test = dfs_to_svd_input(dfs, 'test', cols, norm_dims, n=n)
    mean,std,_ = get_normalized_coefs(svd_train,traj_est,predef_mean_std_tuple=None)
    prepare_and_dump_jsons(svd_train, traj_est, data_dir, split='train', out_dir_base=base_dir,resize_dim=resize_dim,crop_dims = crop_dims,predef_mean_std_tuple=None)
    prepare_and_dump_jsons(svd_val, traj_est, data_dir, split='val', out_dir_base=base_dir,resize_dim=resize_dim,crop_dims = crop_dims,predef_mean_std_tuple=(mean,std))
    prepare_and_dump_jsons(svd_test, traj_est, data_dir, split='test', out_dir_base=base_dir,resize_dim=resize_dim, crop_dims=crop_dims,predef_mean_std_tuple=(mean,std))

#base_dir = '/home/madr/Projects/RK/Imitation_learning/Viper_data'
#data_dir = os.path.join(base_dir, 'data')
#cols = [" camX", "camY"]
#norm_dims = [0, 1]
def execute(cmd):
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True) as proc:
        for stdout_line in iter(proc.stdout.readline, ""):
            yield stdout_line

def get_model_output_dir(base_dir,n,nkey,resize_dim):
    return f"{base_dir}/test_modelsn{n}nkey{nkey}resize_dim{resize_dim[0]}x{resize_dim[1]}"


#DOESNT WORK ON MASTER BRANCH
def routine(n, nkey,resize_dim,dim,id,crop_dims):
    result_df = pd.DataFrame(columns=["n", "nkey", "resize_H", "resize_W", "coefs", "euclid", "proj"])
    load_treat_and_jsonize(base_dir,data_dir,cols,norm_dims,n,nkey,resize_dim=resize_dim,crop_dims = crop_dims)
    dfs = load_data(data_dir, 'train')
    svd_train = dfs_to_svd_input(dfs, 'train', cols, norm_dims, n=n)
    traj_est = TrajectoryEstimator(svd_train[0])
    traj_est.calc_basisFcns(threshold=None,nc=nkey)
    model_output_dir = get_model_output_dir(base_dir,n,nkey,resize_dim)
    model_dict = {'num_key_points':nkey , 'input_shape' : [resize_dim[0],resize_dim[1],3]}
    trainer_dir = os.path.join(os.path.join(os.getcwd()),"C073_Robotcell_keypoints","tf_model1","tf_model1")
    model_dict_s = json.dumps(model_dict)
    shell_str = ['python',os.path.join(trainer_dir,'train.py'),
              f"--train_dir=/home/madr/Projects/RK/Imitation_learning/Viper_data/fits_nc{nkey}_steps{n}_size{resize_dim[0]}_dim{dim}/annotations/train",
              f"--valid_dir=/home/madr/Projects/RK/Imitation_learning/Viper_data/fits_nc{nkey}_steps{n}_size{resize_dim[0]}_dim{dim}/annotations/val",
              "--model_name=Model_RK2021_lego_coefs",
              "--num_epochs=5000",
              "--label_type=RAW",
              "--batch_size=6",
              f"--output_dir={model_output_dir}",
              f"--model_arg_dict={model_dict_s}",
              ]
    for x in execute(shell_str):
        print(x)
    time.sleep(3)

    tester = ModelTester(model_name='Model_RK2021_lego_coefs', model_dir=model_output_dir,gpu_mem_frac=0.2,input_shape =[resize_dim[0],resize_dim[1],3] ,num_key_points=nkey)
    files = np.load("/home/madr/Projects/RK/Imitation_learning/Viper_data/test_models/train_history.npz")
    val_loss = np.min(files['valid_loss_history'])
    file_pairs = get_file_pairs(data_dir,'val')
    pred_coefs = []
    for id, file_pair in file_pairs.items():
        jpg_file, txt_file = file_pair
        img = cv2.imread(os.path.join(data_dir, 'val', jpg_file))
        img = img[:450, 450:1130]
        img = cv2.resize(img, resize_dim)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#        pred_coefs.append(tester.get_key_points_raw([img]))
        pred_coefs.append(tester.get_key_points_raw([img_rgb]))
    pred_coefs = np.array(pred_coefs).squeeze()

    traj_est = traj_est_from_data_dir(data_dir,'train',cols,norm_dims, n,nkey)
    dfs = load_data(data_dir,'val')
    svd_input_coarse,_ = dfs_to_svd_input(dfs,'val',cols,norm_dims,n)
    svd_input_fine,_ = dfs_to_svd_input(dfs,'val',cols,norm_dims,n)
    reshaped_data = traj_est.create_reformed_data(svd_input_coarse)
    print("RESHAPED DATA",reshaped_data.shape)
    val_coefs = np.zeros((reshaped_data.shape[0],nkey))
    for i in range(svd_input_coarse.shape[0]):
        to_insert = reshaped_data.transpose()[:,i]
        new_var = np.array(traj_est.fit_leastsq(to_insert)[0])
        val_coefs[i,:] = new_var
    res = validater.full_validation(traj_est, pred_coefs.transpose(), val_coefs.transpose(), svd_input_fine)
    res_mean = [np.mean(vals) for vals in res.values()]
    res_ls = [n, nkey, resize_dim[0], resize_dim[1]] + list(res_mean)
    result_dict = {x: y for x, y in zip(result_df.columns, res_ls)}
    result_df.sort_values(by=['euclid'], ascending=False, inplace=True)
    result_df.to_csv(os.path.join(base_dir, "result.csv"))
    result_df = result_df.append(result_dict,ignore_index=True)
    print(result_df)
# id = 0
# n= 50
# nkey = 8
# dim = 2
# resize_h = 50
# #res = routine(n,nkey,(resize_h,resize_h),2,id)
#
# dfs = load_data(data_dir,'train')
# data,indices = dfs_to_svd_input(dfs,'train',cols,norm_dims,n)
# traj_est = traj_est_from_data_dir(data_dir, 'train', cols, norm_dims, n, nkey)
# with open("/home/madr/Projects/RK/Imitation_learning/Viperkode/C073_Robotcell_keypoints/tf_model1/tf_model1/RK2021_folder/basis_fn.npy" , 'wb') as f:
#     np.save(f, traj_est.basisFcns)
# data_reshaped = traj_est.create_reformed_data(data)
# traj_est.calc_basisFcns(threshold=None,nc=nkey)
# #tester = ModelTester(model_name='Model_RK2021_lego_coefs', model_dir=get_model_output_dir(base_dir,n,nkey,resize_dim=(resize_h,resize_h)), gpu_mem_frac=0.2,
# #                     input_shape=[resize_h, resize_h, 3], num_key_points=nkey)
# tester = ModelTester( model_name = 'Model_RK2021_lego_coefs', model_dir = "/home/madr/Projects/RK/Imitation_learning/Viperkode/C073_Robotcell_keypoints/tf_model1/tf_model1/models/Model_RK2021_lego_coefs_b6_e5000_lr0.0001", gpu_mem_frac=0.2)
#
# #get pred coefs
# data_ind = 41
# dfs = load_data(data_dir,'val')
# data,indices = dfs_to_svd_input(dfs,'val',cols,norm_dims,n)
# ind = indices.index(data_ind)
# data_reshaped = traj_est.create_reformed_data(data)
# coefs = traj_est.fit_leastsq(data_reshaped[ind])[0]
# trajs = np.matmul(traj_est.basisFcns,coefs)
# jpg, txt = get_file_pairs(data_dir,'val')[data_ind]
# img = cv2.imread(os.path.join(data_dir, 'val', jpg))
# img = img[:450, 450:1130]
# img = cv2.resize(img, (resize_h,resize_h))
# img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# pred_coefs = tester.get_key_points_raw([img_rgb])[0]
# pred_trajs = np.matmul(traj_est.basisFcns,pred_coefs)
# x = traj_est.trajectory(coefs)
# plt.plot(data[ind,:,0],data[ind,:,1])
# plt.plot(x[0:50],x[50:])
# plt.plot(pred_trajs[:50],pred_trajs[50:])

#y = torch.cat([torch.zeros((5,5)),torch.arange(0,25,1).reshape(5,5),torch.zeros((5,5))],axis=1)
#x = torch.arange(0,36).reshape(6,6)
#x.transpose(0,1).reshape(n_split,dim_r//n_split,dim_r).transpose(1,2)

#train_files = get_file_pairs(data_dir,'train')
#val_files = get_file_pairs(data_dir,'val')
#test_files = get_file_pairs(data_dir,'test')
#files = [train_files.keys(),val_files.keys(),test_files.keys()]
#files = [set(list(ls)) for ls in files]
#files[0].intersection(files[1]).intersection(files[2])
#set(range(0,251))-files[0].union(files[1]).union(files[2])