import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from scipy import stats

from trajectory_estimator import TrajectoryEstimator    
import glob
import json
import sys
import os
import cv2
from scipy.optimize import leastsq
from shutil import copy



def resample(x_in, y_in, n):

    # interpolate between data points
    N = 10000
    x = np.linspace(x_in[0], x_in[-1], N)
    y = np.interp(x, x_in, y_in)
    
    # resample to n points (equidistant along path)
    xd = np.diff(x)
    yd = np.diff(y)
    dist = np.sqrt(xd**2+yd**2)
    u = np.cumsum(dist)
    u = np.hstack([[0],u])
    t = np.linspace(0,u.max(),n)
    xn = np.interp(t, u, x)
    yn = np.interp(t, u, y)

    return xn, yn



def plot_sample(dst_path, image, anno=None, pred=None, anno_label='annotation', pred_label='prediction'):
    fig, axis = plt.subplots(1,1)
    axis.imshow(image)
    colours_anno = 100*['red']
    colours_pred = 100*['dodgerblue']
    for i in range(len(anno)):
        anno_obj = axis.scatter(anno[i][0], anno[i][1], marker='o', s=15, c=colours_anno[i])
        axis.plot([x for (x,y) in anno], [y for (x,y) in anno], c=colours_anno[0])
    anno_obj.set_label(anno_label)
    for i in range(len(pred)):
        pred_obj = axis.scatter(pred[i][0], pred[i][1], marker='x', s=15, c=colours_pred[i])
        axis.plot([x for (x,y) in pred], [y for (x,y) in pred], c=colours_pred[0])
    pred_obj.set_label(pred_label)
    axis.legend()
    axis.axis(xmin=0, xmax=image.shape[1], ymax=0, ymin=image.shape[0])
    fig.savefig(dst_path)
    plt.close(fig)



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


def read_json(fname):
    with open(fname) as f:
        data = json.load(f)
    points = data['shapes'][0]['points']
    return np.array(points)




        
if __name__ == '__main__':

    
    n_step = 50
    do_resample = True

    outdir_resample = './keypoint_annotations/resample_steps'+str(n_step)
    if not os.path.isdir(outdir_resample):
        os.mkdir(outdir_resample)    
        os.mkdir(outdir_resample+'/train')
        os.mkdir(outdir_resample+'/test')
        os.mkdir(outdir_resample+'/val')

    files = glob.glob('./keypoint_annotations/train/*.json')
    files.sort()
    data_arr = []
    for fname in files:
        print(fname)
        f = open(fname)
        data = json.load(f)
        f.close()
        points = data['shapes'][0]['points']
        x_ = [xi for xi,yi in points]
        y_ = [yi for xi,yi in points]
        x_ = np.array(x_)
        y_ = np.array(y_)
        if do_resample:
            x, y = resample(x_, y_, n_step)
        else:
            x, y = x_, y_
            n_step = len(points)
        pts = list(zip(x,y))
        data_arr.append(pts)
        data['shapes'][0]['points'] = pts
        with open(outdir_resample + '/train/' + os.path.basename(fname), "w") as fout:
            json.dump(data, fout, indent=2)
        copy(os.path.splitext(fname)[0]+'.jpg', outdir_resample+'/train/'+os.path.splitext(os.path.basename(fname))[0]+'.jpg')
        
        
    data_arr = np.array(data_arr)
    
    # perform SVD
    traj_est = TrajectoryEstimator(data_arr)
    data_arr = np.copy(traj_est.data) #why?
    traj_est.calc_basisFcns(threshold=None, nc=18)
    print('Diagonal:')
    print('---------')
    print(traj_est.S)
    print('')
    print('Using nc='+str(traj_est.nc))
    print('')
    
    

    
    # create output directory
    outdir = './fits_nc'+str(traj_est.nc)+'_steps'+str(n_step)+('_resample' if do_resample else '')
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
        os.mkdir(outdir+'/plots')
        os.mkdir(outdir+'/plots/train')
        os.mkdir(outdir+'/plots/val')
        os.mkdir(outdir+'/plots/test')
        os.mkdir(outdir+'/annotations')
        os.mkdir(outdir+'/annotations/train')
        os.mkdir(outdir+'/annotations/val')
        os.mkdir(outdir+'/annotations/test')
        os.mkdir(outdir+'/predictions')
        os.mkdir(outdir+'/predictions/train')
        os.mkdir(outdir+'/predictions/val')
        os.mkdir(outdir+'/predictions/test')

    np.save(outdir+'/basisFcns.npy', traj_est.basisFcns)


    # plot + json (train)
    for idx in range(len(files)):
        fname = files[idx][0]
        name = fname
        print(name)
        img_name = fname[:-5]+'.jpg'
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        if image is None:
            sys.exit('could not read image: '+img_name)
        target = data_arr[idx]
        result = traj_est.fit_leastsq(target)
        est_target = traj_est.trajectory(result[0])
        # print((traj_est1.VT[:,idx])[:traj_est1.nc])
        #anno = list(zip(data_arr[idx][:n_step], data_arr[idx][n_step:2*n_step]))
        #fit  = list(zip(est_target[:n_step], est_target[n_step:2*n_step]))
        #plot_sample(dst_path=outdir+'/plots/train/'+name+'.jpg', image=image, anno=anno, pred=fit, anno_label='annotation', pred_label='fit (train)')
        coefs = result[0]
        dump_json(outdir+'/annotations/train/'+name+'.json', image_path=name+'.jpg', image_height=image.shape[0], image_width=image.shape[1], points=coefs, label='coefs', shape_type='polygon')
        copy(img_name, outdir+'/annotations/train/'+name+'.jpg')



        

    #------------------
    # Validation data
    #------------------
    files = glob.glob('./keypoint_annotations/val/*.json')
    files.sort()
    data_arr = []   
    c = 0
    for fname in files:
        with open(fname) as f:
            data = json.load(f)
        points = data['shapes'][0]['points']
        x_ = [xi for xi,yi in points]
        y_ = [yi for xi,yi in points]
        x_ = np.array(x_)
        y_ = np.array(y_)
        if do_resample:
             x, y = resample(x_, y_, n_step)
        else:
            x, y = x_, y_
            n_step = len(points)
        pts = list(zip(x,y))       
        data_arr.append(pts)
        data['shapes'][0]['points'] = pts
        with open(outdir_resample + '/val/' + os.path.basename(fname), "w") as fout:
            json.dump(data, fout, indent=2)
        copy(os.path.splitext(fname)[0]+'.jpg', outdir_resample+'/val/'+os.path.splitext(os.path.basename(fname))[0]+'.jpg')

    data_arr = np.array(data_arr)
    traj_est_val = TrajectoryEstimator(data_arr)
    data_arr = np.copy(traj_est_val.data)

    for idx in range(len(files)):
        fname = files[idx]
        name = os.path.splitext(os.path.basename(fname))[0]
        print(name)
        img_name = fname[:-5]+'.jpg'
        image = cv2.imread(img_name) 
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        if image is None:
            sys.exit('could not read image: '+img_name)
        target = data_arr[idx]
        result = traj_est.fit_leastsq(target)
        est_target = traj_est.trajectory(result[0])
        anno = list(zip(data_arr[idx][:n_step], data_arr[idx][n_step:2*n_step]))
        fit  = list(zip(est_target[:n_step], est_target[n_step:2*n_step]))
        plot_sample(dst_path=outdir+'/plots/val/'+name+'.jpg', image=image, anno=anno, pred=fit, anno_label='annotation', pred_label='fit (val)')
        coefs = result[0]
        dump_json(outdir+'/annotations/val/'+name+'.json', image_path=name+'.jpg', image_height=image.shape[0], image_width=image.shape[1], points=coefs, label='coefs', shape_type='polygon')
        copy(img_name, outdir+'/annotations/val/'+name+'.jpg')



    #------------------
    # Test data
    #------------------
    files = glob.glob('./keypoint_annotations/test/*.json')
    files.sort()
    data_arr = []   
    c = 0
    for fname in files:
        with open(fname) as f:
            data = json.load(f)
        points = data['shapes'][0]['points']
        x_ = [xi for xi,yi in points]
        y_ = [yi for xi,yi in points]
        x_ = np.array(x_)
        y_ = np.array(y_)
        if do_resample:
             x, y = resample(x_, y_, n_step)
        else:
            x, y = x_, y_
            n_step = len(points)
        pts = list(zip(x,y))       
        data_arr.append(pts)
        data['shapes'][0]['points'] = pts
        with open(outdir_resample + '/test/' + os.path.basename(fname), "w") as fout:
            json.dump(data, fout, indent=2)
        copy(os.path.splitext(fname)[0]+'.jpg', outdir_resample+'/test/'+os.path.splitext(os.path.basename(fname))[0]+'.jpg')

    data_arr = np.array(data_arr)
    traj_est_test = TrajectoryEstimator(data_arr)
    data_arr = np.copy(traj_est_test.data)

    for idx in range(len(files)):
        fname = files[idx]
        name = os.path.splitext(os.path.basename(fname))[0]
        print(name)
        img_name = fname[:-5]+'.jpg'
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        if image is None:
            sys.exit('could not read image: '+img_name)
        target = data_arr[idx]
        result = traj_est.fit_leastsq(target)
        est_target = traj_est.trajectory(result[0])
        anno = list(zip(data_arr[idx][:n_step], data_arr[idx][n_step:2*n_step]))
        fit  = list(zip(est_target[:n_step], est_target[n_step:2*n_step]))
        plot_sample(dst_path=outdir+'/plots/test/'+name+'.jpg', image=image, anno=anno, pred=fit, anno_label='annotation', pred_label='fit (test)')
        coefs = result[0]
        dump_json(outdir+'/annotations/test/'+name+'.json', image_path=name+'.jpg', image_height=image.shape[0], image_width=image.shape[1], points=coefs, label='coefs', shape_type='polygon')
        copy(img_name, outdir+'/annotations/test/'+name+'.jpg')








