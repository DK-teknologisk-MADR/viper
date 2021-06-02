import validater
import numpy as np
from numba import vectorize
import matplotlib.pyplot as plt
l = 1
n = 500
split_nr = 4 * n
gt_split_nr = 4 * 2 * n
dim = 2


def sq_curve(l,n):
    #starting from line from 4th to 1st quadrant:
    ln41 = np.linspace(np.array([l,-l]),np.array([l,l]),n+1)[:-1]
    ln12 = np.linspace(np.array([l,l]),np.array([-l,l]),n+1)[:-1]
    ln23 = np.linspace(np.array([-l,l]),np.array([-l,-l]),n+1)[:-1]
    ln34 = np.linspace(np.array([-l,-l]),np.array([l,-l]),n+1)[:-1]
    return np.concatenate([ln41,ln12,ln23,ln34])

@vectorize
def antideriv_hypotenuse(x):
    return (1/2) * (x * np.sqrt(1+x**2) + np.arcsinh(x))

pred = np.zeros((3,split_nr,dim))
data = np.zeros((3,gt_split_nr,dim))
for i in range(3):
    data[i] = sq_curve(2*l,2 * n)
    pred[i] = sq_curve(l,n)
x = validater.validate(data, pred, 2)
theo_result = (8 + 8 * antideriv_hypotenuse(1))/(4*2*2*l)
print(f"algorithm got {x}\ntrue value is {theo_result}\n"
      f"difference is{x-theo_result}")
corner_dists,proj_is_valid,projs = validater.gpu_comp(data, pred, obs_nr, split_nr, gt_split_nr, dim)
x*8
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


obs_nr = 3
#SINUS APPROX CURVE:
pred = np.zeros((obs_nr,split_nr,dim))
for i in range(obs_nr):
    x = np.linspace(0,5,split_nr)
    pred[i] = np.array([x,1.5 * np.sin(x )+np.random.normal(0,0.3,split_nr)]).T
data = np.zeros((obs_nr,gt_split_nr,dim))
for i in range(obs_nr):
    x = np.linspace(0,5,gt_split_nr)
    data[i] = np.array([x,1.5 * np.sin(x )]).T

corner_dists,proj_is_valid,projs = validater.gpu_comp(data, pred, obs_nr, split_nr, gt_split_nr, dim)
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
