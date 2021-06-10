import cv2
import validater
import os
import sys
import numpy as np
from trajectory_estimator import TrajectoryEstimator
from toy_file import get_file_pairs, load_data, dfs_to_svd_input, resample
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.getcwd(),'C073_Robotcell_keypoints/tf_model1/tf_model1'))
from model_tester import ModelTester
base_dir = '/home/madr/Projects/RK/Imitation learning/Viper_data'
data_dir = os.path.join(base_dir,'data')
file_pairs = get_file_pairs(data_dir,'val')
tester = ModelTester(model_name = 'Model_RK2021_lego_coefs', model_dir = os.path.join(os.getcwd(),'C073_Robotcell_keypoints/tf_model1/tf_model1/models/Model_RK2021_lego_coefs_b8_e8000_lr0.0001'))


cols = [" camX", "camY"," camZ"]
norm_dims = [0,1,2]
n = 50

dfs = load_data(data_dir,'train')
svd_train = dfs_to_svd_input(dfs,'train',cols,norm_dims,n=n)
del dfs
traj_est = TrajectoryEstimator(svd_train[0])
traj_est.calc_basisFcns(None,11)

obs = 160
fig = plt.figure()
ax = fig.gca(projection='3d')
svd_train_resh=traj_est.create_reformed_data(svd_train[0])
svd_curves = []
for i in range(svd_train[0].shape[0]):
    target = svd_train_resh[i]
    result = traj_est.fit_leastsq(target)
    est_target = traj_est.trajectory(result[0])
    svd_curves.append(est_target)
svd_curves = np.array(svd_curves)
svd_curves = traj_est.reform_to_std_data_shape(svd_curves)
#coarse_curve_gt = traj_est.reform_to_std_data_shape(traj_est.trajectory(pred_coefs.transpose()))

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot3D(svd_train[0][obs,:,0],svd_train[0][obs,:,1],svd_train[0][obs,:,2])
ax.plot3D(svd_curves[obs,:,0],svd_curves[obs,:,1],svd_curves[obs,:,2])
svd_train[0][160]
dfs_val = load_data(data_dir,'val')
i = list(dfs_val.keys())[0]
gt_data,gt_indices = dfs_to_svd_input(dfs_val,'val',cols,norm_dims,n=1000)
svd_val,indices = dfs_to_svd_input(dfs_val,'val',cols,norm_dims,n=n)
svd_val_reshaped = traj_est.create_reformed_data(svd_val)
gt_coefs = []
for obs in svd_val_reshaped:
    coef=traj_est.fit_leastsq(obs)
    gt_coefs.append(coef[0])
gt_coefs = np.array(gt_coefs)
imgs = []
pred_coefs = []
for id,file_pair in file_pairs.items():
    jpg_file,txt_file = file_pair
    img = cv2.imread(os.path.join(data_dir,'val',jpg_file))
    img = img[:450, 450:1130]
    img = cv2.resize(img, (226, 150))
    pred_coefs.append( tester.get_key_points_raw([img]) )
pred_coefs = np.array(pred_coefs).squeeze()
validater.full_validation(traj_est,pred_coefs.transpose(),gt_coefs.transpose(),gt_data)


pred_curve = traj_est.reform_to_std_data_shape(traj_est.trajectory(gt_coefs.transpose()))
coarse_curve_gt = traj_est.reform_to_std_data_shape(traj_est.trajectory(pred_coefs.transpose()))
obs = 2
ax.plot3D(pred_curve[obs,:,0],pred_curve[obs,:,1],pred_curve[obs,:,2])
ax.plot3D(svd_val[obs,:,0],svd_val[obs,:,1],svd_val[obs,:,2])
ax.plot3D(coarse_curve_gt[obs,:,0],coarse_curve_gt[obs,:,1],coarse_curve_gt[obs,:,2])
