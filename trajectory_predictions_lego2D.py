import numpy as np
import matplotlib.pyplot as plt
import random
#from mpl_toolkits.mplot3d import Axes3D
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



def dump_json(output_path, image_path, image_height, image_width, points, label, shape_type):
    assert type(points) == np.ndarray
    data = dict()
    data['imagePath'] = image_path
    data['imageData'] = None
    shape_entry = dict()
    shape_entry['label'] = 'polygon'
    shape_entry['shape_type'] = 'polygon'
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


    INPUT = 'train'    
    n_step = 50
    outdir = './fits_nc18_steps'+str(n_step)+'_resample'

    basisFcns = np.load(outdir+'/basisFcns.npy')
    print(basisFcns.shape)

    traj_est = TrajectoryEstimator(np.array([[[0]]]))
    traj_est.basisFcns = basisFcns
    
    predictions = glob.glob(outdir+'/predictions/'+INPUT+'/*.json')
    predictions.sort()
    for idx in range(len(predictions)):
        fname = predictions[idx]
        name = os.path.splitext(os.path.basename(fname))[0]
        print(name)
        img_name = fname[:-5]+'.jpg'
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        if image is None:
            sys.exit('could not read image: '+img_name)
        fit_coefs   = read_json(outdir+'/annotations/'+INPUT+'/'+name+'.json')
        est_target  = traj_est.trajectory(fit_coefs)
        pred_coefs  = read_json(fname)
        pred_target = traj_est.trajectory(pred_coefs)    
        fit  = list(zip(est_target[:n_step], est_target[n_step:2*n_step]))
        pred = list(zip(pred_target[:n_step], pred_target[n_step:2*n_step]))
        plot_sample(dst_path=fname[:-5]+'__pred.jpg', image=image, anno=fit, pred=pred, anno_label='fit', pred_label='prediction')

