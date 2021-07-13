import cv2
import validater
import os
import sys
import numpy as np
from trajectory_estimator import TrajectoryEstimator
from toy_file import get_file_pairs, load_data, dfs_to_svd_input, resample,data_transform
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.getcwd(),'C073_Robotcell_keypoints/tf_model1/tf_model1'))
from model_tester_RK2021_lego_coefs import ModelTesterRK2021LegoCoefs
base_dir = '/home/madr/Projects/RK/Imitation_learning/Viper_data'
data_dir = os.path.join(base_dir, 'viper_data_june')
file_pairs = get_file_pairs(data_dir,'val')
model_dir = os.path.join(os.getcwd(),'C073_Robotcell_keypoints/tf_model1/tf_model1/models/Model_RK2021_lego_coefs_b16_e200_lr0.0001')
basis_file_path = os.path.join(model_dir,'basis_fcns.npy')
tester = ModelTesterRK2021LegoCoefs(basis_file_path=basis_file_path,model_dir = model_dir, gpu_id=1)


crop_dims = 180,590,420,830
resize_nr = 50
obs_nr = len(file_pairs)
cols = [" camX", "camY"," camZ"]
norm_dims = [0,1,2]
n = 50
dfs = load_data(data_dir,'train')
dim=len(cols)
svd_train = dfs_to_svd_input(dfs,'train',cols,norm_dims,n=n)
del dfs

n_kpts = 12
traj_est = TrajectoryEstimator(svd_train[0])
traj_est.calc_basisFcns(None,n_kpts)
loaded_basis_file_path = np.load(basis_file_path)


np.save(os.path.join(model_dir,"basis_fcns.npy"),traj_est.basisFcns)
indices = svd_train[1]
obs = 0

dfs_val = load_data(data_dir,'val')
gt_data,gt_indices = dfs_to_svd_input(dfs_val,'val',cols,norm_dims,n=1000)
svd_val,indices = dfs_to_svd_input(dfs_val,'val',cols,norm_dims,n=n)
svd_val_reshaped = traj_est.create_reformed_data(svd_val)
pic_index = indices[0]
img_name = file_pairs[pic_index][0]
img = cv2.imread(os.path.join(data_dir, 'val', img_name))
img = data_transform(img, (resize_nr, resize_nr), crop_dims)
cv2.imshow("lol",img)
fig = plt.figure()
#ax = fig.gca(projection='32')
svd_train_resh=traj_est.create_reformed_data(svd_train[0])
svd_curves = []
pred_coefs = np.zeros((obs_nr,1,n_kpts))
pred_curve = np.zeros((obs_nr,n,dim))
for i in range(obs_nr):
    target = svd_val_reshaped[i]
    result = traj_est.fit_leastsq(target)
    est_target = traj_est.trajectory(result[0])
    svd_curves.append(est_target)
    pic_index = indices[i]
    img_name = file_pairs[pic_index][0]
    img = cv2.imread(os.path.join(data_dir,'val',img_name))
    img = data_transform(img,(resize_nr,resize_nr),crop_dims)
    pred_coefs[i] = tester.get_coefs([img])
    pred_curve[i] = (tester.get_key_points_raw([img]).reshape(dim,n).transpose())
print(pred_coefs)
plt.plot(svd_val[10, :, 0], svd_val[10, :, 1])
plt.plot(svd_curves[12, :, 0], svd_curves[10, :, 1])
plt.plot(pred_curve[12, :, 0], pred_curve[12, :, 1])
plt.xlim(-0.7, -0.22)
plt.ylim(-0.55, -0.24)
#    tester(img_name)
svd_curves = np.array(svd_curves)
svd_curves = traj_est.reform_to_std_data_shape(svd_curves)
#coarse_curve_gt = traj_est.reform_to_std_data_shape(traj_est.trajectory(pred_coefs.transpose()))
os.makedirs(os.path.join(model_dir,"plots"),exist_ok=True)
dim = 2
if dim == 2:
    for i in range(obs_nr):
        fig = plt.figure()
        plt.plot(svd_val[i,:,0],svd_val[i,:,1])
        plt.plot(svd_curves[i,:,0],svd_curves[i,:,1])
        plt.plot(pred_curve[i,:,0],pred_curve[i,:,1])
        plt.xlim(-0.7,-0.22)
        plt.ylim(-0.55,-0.24)
        plt.savefig(os.path.join(model_dir,"plots",str(indices[i])),dpi=200)
        plt.close(fig)
if dim==3:
    for i in range(obs_nr):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot3D(svd_val[i, :, 0], svd_val[i, :, 1],svd_val[i, :, 2])
        ax.plot3D(svd_curves[i, :, 0], svd_curves[i, :, 1],svd_curves[i, :, 2])
        ax.plot3D(pred_curve[i, :, 0], pred_curve[i, :, 1],pred_curve[i, :, 2])
        plt.savefig(os.path.join(model_dir,"plots",str(indices[i])),dpi=200)
        plt.close(fig)


#fig = plt.figure()
#ax = fig.gca(projection='3d')
# plt.plot(svd_train[0][obs,:,0],svd_train[0][obs,:,1],svd_train[0][obs,:,2])
# ax.plot3D(svd_train[0][obs,:,0],svd_train[0][obs,:,1],svd_train[0][obs,:,2])
# ax.plot3D(svd_curves[obs,:,0],svd_curves[obs,:,1],svd_curves[obs,:,2])
# svd_train[0][160]
# dfs_val = load_data(data_dir,'val')
# i = list(dfs_val.keys())[0]
# gt_data,gt_indices = dfs_to_svd_input(dfs_val,'val',cols,norm_dims,n=1000)
# svd_val,indices = dfs_to_svd_input(dfs_val,'val',cols,norm_dims,n=n)
# svd_val_reshaped = traj_est.create_reformed_data(svd_val)
# gt_coefs = []
# for obs in svd_val_reshaped:
#     coef=traj_est.fit_leastsq(obs)
#     gt_coefs.append(coef[0])
# gt_coefs = np.array(gt_coefs)
# imgs = []
# pred_coefs = []
# for id,file_pair in file_pairs.items():
#     jpg_file,txt_file = file_pair
#     img = cv2.imread(os.path.join(data_dir,'val',jpg_file))
#     img = img[:450, 450:1130]
#     img = cv2.resize(img, (226, 150))
#     pred_coefs.append( tester.get_key_points_raw([img]) )
# pred_coefs = np.array(pred_coefs).squeeze()
# validater.full_validation(traj_est,pred_coefs.transpose(),gt_coefs.transpose(),gt_data)
#
#
# pred_curve = traj_est.reform_to_std_data_shape(traj_est.trajectory(gt_coefs.transpose()))
# coarse_curve_gt = traj_est.reform_to_std_data_shape(traj_est.trajectory(pred_coefs.transpose()))
# obs = 2
# ax.plot3D(pred_curve[obs,:,0],pred_curve[obs,:,1],pred_curve[obs,:,2])
# ax.plot3D(svd_val[obs,:,0],svd_val[obs,:,1],svd_val[obs,:,2])
# ax.plot3D(coarse_curve_gt[obs,:,0],coarse_curve_gt[obs,:,1],coarse_curve_gt[obs,:,2])
