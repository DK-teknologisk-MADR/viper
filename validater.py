import  torch
import numpy as np
from numba import njit, prange


def gpu_comp(data,pred,obs_nr,split_nr,gt_split_nr,dim):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ts = torch.tensor(pred,device=device)
    dts = ( ts[:,1:,:]-ts[:,:(split_nr-1),:] ).reshape(obs_nr*(split_nr-1),dim)
    ts_wo_last = ts[:,:(split_nr-1),:].reshape(obs_nr*(split_nr-1),dim)
    ts_wo_first = ts[:, 1:].reshape(obs_nr * (split_nr - 1), dim)
    proj_mats = torch.bmm(dts.unsqueeze(2),dts.unsqueeze(1)) / torch.bmm(dts.unsqueeze(1),dts.unsqueeze(2))
    ts_data = torch.tensor(data)
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
    obs_nr,gt_split_nr,dim = data.shape
    obs_per_batch = np.ceil(obs_nr/batches).astype(np.int)
    indices = np.arange(0,obs_nr+1,obs_per_batch)
    res_ls = []
    for i in range(0,batches-1):
        res_ls.append(validate_batch(data[indices[i]:indices[i+1]], pred[indices[i]:indices[i+1]]))
    if not indices[-1] == obs_nr:
        res_ls.append(validate_batch(data[indices[-1]:obs_nr], pred[indices[-1]:obs_nr]))
    res_ls = np.hstack(res_ls)
    return res_ls


