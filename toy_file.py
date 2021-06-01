from trajectory_estimator import TrajectoryEstimator
import os
import json
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
data_dir = os.getcwd()
#traj_est = TrajectoryEstimator()
data_dir = '/home/madr/Projects/RK/Imitation learning/Viper_data/data'

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

def load_data(data_dir,cols,split):
    dfs = []
    file_pairs = get_file_pairs(data_dir,split)
    for i,files in enumerate(file_pairs.items()):
        key, ls = files
        df = pd.read_csv(os.path.join(data_dir,split, ls[1] ) )
        if all(isinstance(col,str) for col in cols):
            df = df.loc[:,cols].values
        elif all(isinstance(col,int) for col in cols):
            df = df.iloc[:,cols].values
        else:
            raise ValueError("cols should be a list of either ints or strings")
        nan_mask = df.isna()
        if np.any(nan_mask.any(axis=1)):
            df = df.dropna()
            print(f'dropped {nan_mask.any(axis=1).sum()} rows due to NaNs in data indexed {i} named {ls[1]}')
        dfs.append(df)
    return dfs

def get_svd_input(data_dir,split,cols,dims_to_normalize,n=60):
    data_ls = load_data(data_dir,cols,split)
    dims_n = len(cols)
    obs_n = len(data_ls)
    svd_input = np.zeros((obs_n,n,dims_n))
    for i,curve_dt in enumerate ( data_ls ):
        svd_input[i,:] = resample(curve_dt, n, dims_to_normalize)
    return svd_input
dfs = load_data(data_dir,[" camX", "camY",],'train')
svd_input = get_svd_input(data_dir,'train',[" camX", "camY",],[0,1])
traj_est = TrajectoryEstimator(svd_input)
traj_est.calc_basisFcns(threshold=.90)


svd_input.shape #OK
plt.plot(svd_input[75,:,0],svd_input[75,:,1]) #OK
np.where(np.isnan(svd_input))

