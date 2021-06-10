import  torch
import numpy as np
from numba import njit, prange
import sys
import os

def gpu_comp(data,pred,obs_nr,split_nr,gt_split_nr,dim):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ts = torch.tensor(pred,device=device)
    dts = ( ts[:,1:,:]-ts[:,:(split_nr-1),:] ).reshape(obs_nr*(split_nr-1),dim)
    ts_wo_last = ts[:,:(split_nr-1),:].reshape(obs_nr*(split_nr-1),dim)
    ts_wo_first = ts[:, 1:].reshape(obs_nr * (split_nr - 1), dim)
    proj_mats = torch.bmm(dts.unsqueeze(2),dts.unsqueeze(1)) / torch.bmm(dts.unsqueeze(1),dts.unsqueeze(2))
    ts_data = torch.tensor(data,device=device)
    print(ts.device,ts_wo_last.device,ts_wo_first.device,proj_mats.device,ts_data.device)
    dat_wo_last = ts_data.unsqueeze(1).expand(obs_nr,split_nr-1,gt_split_nr,dim).reshape(-1,gt_split_nr,dim).transpose(1,2)
    projs =ts_wo_last[:,:,None] + torch.bmm(proj_mats, dat_wo_last-ts_wo_last[:,:,None])

    legit_projs_mask_pre = torch.maximum( torch.abs(ts_wo_last[:,:,None]-projs).sum(axis=1), torch.abs(ts_wo_first[:,:,None]-projs).sum(axis=1))
    legit_projs_mask = legit_projs_mask_pre.le(
        torch.abs(ts_wo_first-ts_wo_last).sum(axis=1)[:,None])

    return (torch.cdist(ts,ts_data,2).min(axis=1)[0].to('cpu').numpy(),
            legit_projs_mask.reshape(obs_nr,split_nr-1,gt_split_nr).to('cpu').numpy(),
            projs.reshape(obs_nr,split_nr-1,dim,gt_split_nr).transpose(2,3).to('cpu').numpy(),
    )

@njit(parallel=True)
def dist_proj_data(data,mask_arr,projs,obs_nr,split_nr,gt_split_nr,dim):
    result = np.full((obs_nr,gt_split_nr),np.inf)
    for obs in prange(obs_nr):
        for pt_ind in prange(gt_split_nr):
            lines = np.full(split_nr,np.inf)
            pt = data[obs,pt_ind,:]
            masks = mask_arr[obs, :, pt_ind]
            for i,mask in enumerate(masks):
                if mask:
                    candidate = projs[obs,i,pt_ind]
                    lines[i] = np.sqrt(((candidate - pt) ** 2).sum())
                    result[obs,pt_ind] = np.min(lines)
    return result

@njit()
def compare_corner_and_projs_then_trapz(data, proj_is_valid, projs,corner_dists,obs_nr,split_nr,gt_split_nr,dim):
    dists = np.minimum(corner_dists, dist_proj_data(data, proj_is_valid, projs, obs_nr, split_nr, gt_split_nr, dim))
    integral = np.trapz(dists,dx=1/gt_split_nr)
    return integral

def validate_batch(data,pred):
    obs_nr,gt_split_nr,dim = data.shape
    split_nr = pred.shape[1]
    corner_dists, proj_is_valid, projs = gpu_comp(data, pred, obs_nr, split_nr, gt_split_nr, dim)
    return compare_corner_and_projs_then_trapz(data,proj_is_valid,projs,corner_dists,obs_nr, split_nr, gt_split_nr, dim)

def validate(data,pred,batches = 1):
    if batches > 1 :
        obs_nr,gt_split_nr,dim = data.shape
        obs_per_batch = np.ceil(obs_nr/batches).astype(np.int)
        indices = np.arange(0,obs_nr+1,obs_per_batch)
        print(indices,obs_nr)
        res_ls = []
        for i in range(0,batches-1):
            res_ls.append(validate_batch(data[indices[i]:indices[i+1]], pred[indices[i]:indices[i+1]]))
        if not indices[-1] == obs_nr:
            res_ls.append(validate_batch(data[indices[-1]:obs_nr], pred[indices[-1]:obs_nr]))
        print(res_ls)
        res = np.hstack(res_ls)
    else:
        res = validate_batch(data,pred)
    return res

def validate_euclid(data_coarse,pred):
    obs_nr,split_nr,dim = data_coarse.shape
    obs_nr,split_nr_two,dim = pred.shape
    assert split_nr == split_nr_two , f"pred split_nr ({split_nr}) should be same as data_coarse split nr ({split_nr_two}). Did you remember to coarse down data?"
    return np.mean(np.linalg.norm(data_coarse-pred,axis=2),axis = 1)

def validate_coef(coef_data,coef_pred):
    return np.mean(np.abs(coef_data-coef_pred),axis=1)


def full_validation(traj_est,coefs_pred, gt_coef, gt_data):
    pred_curve = traj_est.reform_to_std_data_shape(traj_est.trajectory(gt_coef))
    coarse_curve_gt = traj_est.reform_to_std_data_shape(traj_est.trajectory(coefs_pred))
    metrics = {'coefs': validate_coef(gt_coef, coefs_pred),
               'euclid': validate_euclid(pred_curve, coarse_curve_gt),
               'proj': validate(gt_data, pred_curve)}
    return metrics

