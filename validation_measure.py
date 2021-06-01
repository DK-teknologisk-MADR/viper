import numpy as np

def proj_affine(lpt1,lpt2):
    '''
    returns pair to use for affine projection: (Proj mat,translate) so that
    proj(pt) = mat.pt + translate
    '''
    v = lpt2-lpt1
    return ( 1/np.power(v,2) * np.outer(v,v) , lpt1)


def get_pw_proj_funcs(points_ls):
    proj_funcs = []
    for i in range(points_ls[1:]):
        lpt1,lpt2 = points_ls[i-1:i+1]
        proj_funcs[i] = proj_affine(lpt1,lpt2)
