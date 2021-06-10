import os
import json
import time

from validater import validate
import numpy as np
import matplotlib.pyplot as plt
import torch
import subprocess
from trajectory_estimator import TrajectoryEstimator
obs_nr = 100
split_nr = 10
gt_split_nr = 50
dim = 2

pred = np.zeros((obs_nr,split_nr,dim))
data = np.zeros((obs_nr,gt_split_nr,dim))
for i in range(obs_nr):
    x = np.linspace(0,5,gt_split_nr)
    data[i] = np.array([x,1.5 * np.sin(x )]).T
    data[i] += np.array([x,3 * np.cos(x )]).T
    data[i] += np.array([x,4 * np.sqrt(x )]).T
    data[i] += np.array([x,3 * np.tanh(x )]).T

#PSEUDOCODE:
def objective(n_kpts,n_splits,**kpt_kwargs):

    traj_est = TrajectoryEstimator(data)
    traj_est.calc_basisFcns(None,nc=n_kpts)

    curves = traj_est.basisFcns # n_splits X obs_nr
    coefs = traj_est.VT  # obs_nr X n_kpts
    build_and_train_model(gt = coefs, pictures = pictures,n_kpts = n_kpts)
    val_coefs = eval_model(pictures=val_pictures) #  val_obs X n_kpts
    pred = np.matmul(curves,val_coefs)
    return np.mean(validate(data,pred,data.shape[0],n_splits,gt_split_nr,data.shape[2]))

corner_dists,proj_is_valid,projs = main.gpu_comp(data,pred,obs_nr,split_nr,gt_split_nr,dim)
#check
ls = []
for i in range(split_nr-1):
    for j in range(gt_split_nr):
        if proj_is_valid[0][i][j]:
            ls.append(projs[0,i,j,:])
ls = np.array(ls)
plt.gca().set_aspect('equal', adjustable='box')
plt.plot(data[0,:,0],data[0,:,1],'-o')
plt.plot(pred[0,:,0],pred[0,:,1],'-o')
plt.plot(projs[0,0,0,0],projs[0,0,1,0],'-o')
plt.plot(data[0,7,0],data[0,7,1],'-o')
plt.plot(ls[:,0],ls[:,1],'-o',linestyle='None')
