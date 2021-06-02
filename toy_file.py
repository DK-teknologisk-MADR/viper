from trajectory_estimator import TrajectoryEstimator
import os
import json
import numpy as np
import pandas as pd
import re
import cv2
from shutil import copy
import matplotlib.pyplot as plt
data_dir = os.getcwd()
#traj_est = TrajectoryEstimator()
data_dir = '/home/madr/Projects/RK/Imitation learning/Viper_data/data'
base_dir = '/home/madr/Projects/RK/Imitation learning/Viper_data'



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
        curve_dt = df[cols].values
        id_from_index.append(ID)
        svd_input[ind,:] = resample(curve_dt, n, dims_to_normalize)
    return svd_input,id_from_index


dfs = load_data(data_dir,'train')
svd_input,id_from_index = dfs_to_svd_input(dfs,'train',[" camX", "camY",],[0,1],n=50)
traj_est = TrajectoryEstimator(svd_input)
traj_est.calc_basisFcns()
print('Diagonal:')
print('---------')
print(traj_est.S[:traj_est.nc])
print('')
print('Using nc=' + str(traj_est.nc))
print('')
def prepare_and_dump_jsons(svd_input,traj_est,data_dir,split,out_dir_base = base_dir):
    assert isinstance(traj_est,TrajectoryEstimator)
    if out_dir_base is None:
        out_dir_base = os.getcwd()
    outdir = os.path.join(out_dir_base,
                          'fits_nc' + str(traj_est.nc) + '_steps' + str(traj_est.data.shape[1] // traj_est.dim),
                          'annotations',
                          split)
    print(outdir)
    files = get_file_pairs(data_dir,split)
    svd_input_reshaped = np.copy(traj_est.data)  # why?
    print(svd_input.shape[0])
    for idx in range(svd_input.shape[0]):
        id = id_from_index[idx]
        print(files[id][0])
        fname = files[id][0][:-4]
        img_name = files[id][0]
        img_path = os.path.join(data_dir,split,img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        if image is None:
            raise ValueError(f"Could not find image {img_name}")
        target = svd_input_reshaped[idx]
        result = traj_est.fit_leastsq(target)
        est_target = traj_est.trajectory(result[0])
        coefs = result[0]
        print(outdir)
        os.makedirs(outdir, exist_ok=True)
        dump_json(os.path.join(outdir, fname+'.json'), image_path=fname + '.jpg',
                  image_height=image.shape[0], image_width=image.shape[1], points=coefs, label='coefs',
                  shape_type='polygon')
        copy(img_path, os.path.join(outdir, fname+'.jpg'))
prepare_and_dump_jsons(svd_input,traj_est,data_dir,split = 'train')
