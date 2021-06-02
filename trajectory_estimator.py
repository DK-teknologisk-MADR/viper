import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize, curve_fit
from scipy import stats


class TrajectoryEstimator(object):

    def __init__(self, data):
        assert type(data) == np.ndarray
        self.data = None
        # data is an array of demonstrations
        # E.g
        # N demonstrations of k 3D position and orientation:
        # data = [[[x_00, y_00, z_00, ox_00, oy_00, oz_00],
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
        # before SVD, re-arrange into
        # data = [[x_00, ..., x_0k, y_00, y_0k, ..., oz_0k],
        #         ...
        #         [x_N0, ..., x_Nk, y_N0, y_Nk, ..., oz_Nk]],
        print(f'TrajectoryEstimator::__init__() : data size = {len(data)}')
        assert len(data) > 0
        self.data_raw = data
        self.data = self.create_reformed_data(data)
        self.basisFcns  = None
        self.targetTraj = None

    def create_reformed_data(self,data):
        self.dim = data.shape[2]
        data_new = []
        for demo in data:
            d = []
            for i in range(demo.shape[1]):
                d.extend(demo[:,i])
            data_new.append(d)
        data_new       = np.array(data_new)
        return data_new
                
    def calc_basisFcns(self, threshold=0.95, nc=None):
        assert self.data is not None
        print(self.data.shape)
        self.U, self.S, self.VT = np.linalg.svd(self.data.T)
        if threshold is not None:
            trace = np.sum(self.S)
            cumsum = np.cumsum(self.S)
            self.nc = np.sum([1 if cs/trace<threshold else 0 for cs in cumsum])
        elif nc is not None:
            self.nc = nc
        else:
            self.nc = len(self.data[0])
        self.basisFcns = np.matmul(self.U[:,:self.nc], np.diag(self.S[:self.nc]))
        print(self.data.T.shape, self.basisFcns.shape)
              
    def trajectory(self, coefs):
        assert self.basisFcns is not None
        assert len(self.basisFcns.shape) == 2
        assert len(coefs.shape) == 1
        assert self.basisFcns.shape[1] == coefs.shape[0]
        return np.matmul(self.basisFcns, coefs)

    def fit(self, targetTraj, initParams=None):
        if initParams is None:
            initParams = [0.5/self.nc]*self.nc
        def negLogL(par):
            return -np.sum(stats.norm.logpdf(targetTraj.flatten(),self.trajectory(par).flatten()))
        result = minimize(negLogL, initParams, method='Powell')
        return result

    def fit_leastsq(self, targetTraj, initParams=None):
        assert self.basisFcns is not None , "basis_fcns has not yet been calculated"
        def func(xdata, *params):
            return np.matmul(xdata,params)
        result = curve_fit(f=func, xdata=self.basisFcns, ydata=targetTraj, p0=np.zeros((1,self.nc)))
        return result



if __name__ == '__main__':

    print('\nTrajectoryEstimater() :: EXAMPLE\n')
    
    random.seed(314)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    n_demo = 50
    n_step = 100
    tData = np.arange(n_step)/float(n_step)
    data_arr = []

    for i in range(n_demo): 
        C0 = 0.5*(1+random.random())
        C1 = 0.1*random.random()
        C2 = 0.5*(1+random.random())
        C3 = 0.1*random.random()
        C4 = 0.1*random.random()
        xData  = [C0*np.sin(t*np.pi)+C1 for t in tData]
        yData  = [C2*np.sin(t*np.pi)+C3 for t in tData]
        zData  = [t+C4 for t in tData]
        rxData = [C0*np.sin(t*np.pi)+C1 for t in tData]
        ryData = [C2*np.sin(t*np.pi)+C3 for t in tData]
        rzData = [t+C4 for t in tData]
        data = np.concatenate((xData, yData, zData))
        data_arr.append(data)
        ax.plot(xData, yData, zData)

    fig.savefig('trajectory_3D.png', dpi=600)

    traj_est = TrajectoryEstimator(data_arr)
    traj_est.calc_basisFcns(threshold=0.99)
    print('Diagonal:')
    print('---------')
    print(traj_est.S)
    print('')
    print('Using nc='+str(traj_est.nc))
    print('')

    plt.cla()
    for i in range(traj_est.nc):
        coefs = np.zeros(traj_est.nc)
        coefs[i] = 1.0
        bFcn = traj_est.trajectory(coefs)
        ax.plot(bFcn[:n_step], bFcn[n_step:2*n_step], bFcn[2*n_step:])

    fig.savefig('trajectory_3D_basisFcns.png', dpi=600)
    
    idx = 0
    target = data_arr[idx]
    result = traj_est.fit(target)
    est_target = traj_est.trajectory(result.x)
    print(result.x.shape)
    
    print('Estimated Coefficients:')
    print('-----------------------')
    print(result.x)
    print('')
    
    plt.cla()
    ax.plot(data_arr[idx][:n_step], data_arr[idx][n_step:2*n_step], data_arr[idx][2*n_step:])
    ax.plot(est_target[:n_step], est_target[n_step:2*n_step], est_target[2*n_step:])
    fig.savefig('trajectory_3D_fit_demo'+str(idx)+'.png', dpi=600)

    Res = [data_arr[idx][j] - est_target[j] for j in range(n_step*3)]
    Rx  = Res[:n_step]
    R2x = np.square(Rx)
    Ry  = Res[n_step:2*n_step]
    R2y = np.square(Ry)
    Rz  = Res[2*n_step:]
    R2z = np.square(Rz)
    R2  = [R2x[j]+R2y[j]+R2z[j] for j in range(n_step)]
    R   = np.sqrt(R2)

    plt.clf()
    plt.plot(tData, R)
    fig.savefig('trajectory_3D_res_demo'+str(idx)+'.png', dpi=600)
    
    plt.clf()
    plt.plot(tData, Rx)
    fig.savefig('trajectory_3D_resX_demo'+str(idx)+'.png', dpi=600)
    
    plt.clf()
    plt.plot(tData, Ry)
    fig.savefig('trajectory_3D_resY_demo'+str(idx)+'.png', dpi=600)
    
    plt.clf()
    plt.plot(tData, Rz)
    fig.savefig('trajectory_3D_resZ_demo'+str(idx)+'.png', dpi=600)
    
