
import numpy as np
import os
import math
import logging
import random
import datetime
import socket
import pickle
from shapely.geometry import Point

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.pi = torch.acos(torch.zeros(1)).item()*2


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    
def new_log(logdir,filename):
    """Defines logging format.
    """
    filename = os.path.join(logdir,
                            datetime.datetime.now().strftime("log_%Y-%m-%d-%H-%M-%S_"+socket.gethostname()+"_"+filename+".log"))
    logging.basicConfig(level=logging.INFO,
                        filename=filename,
                        format="%(asctime)s - %(name)s - %(message)s",
                        filemode="w")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)   
    
def haversine(input_coords, 
               pred_coords):
    """ Calculate the haversine distances between input_coords and pred_coords.
    
    Args:
        input_coords, pred_coords: Tensors of size (...,N), with (...,0) and (...,1) are
        the latitude and longitude in radians.
    
    Returns:
        The havesine distances between
    """
    R = 6371
    lat_errors = pred_coords[...,0] - input_coords[...,0]
    lon_errors = pred_coords[...,1] - input_coords[...,1]
    a = torch.sin(lat_errors/2)**2\
        +torch.cos(input_coords[:,:,0])*torch.cos(pred_coords[:,:,0])*torch.sin(lon_errors/2)**2
    c = 2*torch.atan2(torch.sqrt(a),torch.sqrt(1-a))
    d = R*c
    return d

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

def top_k_nearest_idx(att_logits, att_idxs, r_vicinity):
    """Keep only k values nearest the current idx.
    
    Args:
        att_logits: a Tensor of shape (bachsize, data_size). 
        att_idxs: a Tensor of shape (bachsize, 1), indicates 
            the current idxs.
        r_vicinity: number of values to be kept.
    """
    device = att_logits.device
    idx_range = torch.arange(att_logits.shape[-1]).to(device).repeat(att_logits.shape[0],1)
    idx_dists = torch.abs(idx_range - att_idxs)
    out = att_logits.clone()
    out[idx_dists >= r_vicinity/2] = -float('Inf')
    return out


# if a value in ndarray is NaN then it will be filled with value of previous row same col
def ffill_roll(arr, fill=0, axis=0):
    mask = np.isnan(arr)
    replaces = np.roll(arr, 1, axis)
    slicing = tuple(0 if i == axis else slice(None) for i in range(arr.ndim))
    replaces[slicing] = fill
    while np.count_nonzero(mask) > 0:
        arr[mask] = replaces[mask]
        mask = np.isnan(arr)
        replaces = np.roll(replaces, 1, axis)
    return arr


def split_into_chunks(out_file, out_dir_, nchunks):
    if not os.path.exists(out_dir_):
        os.makedirs(out_dir_)
        print("==> Directory created {0}".format(out_dir_))
    fpickle = open(out_file, 'rb')
    list_of_dict = pickle.load(fpickle)
    print(list_of_dict.__len__())

    chunksize = int(float(list_of_dict.__len__())/float(nchunks))
    index = 0
    for nc in range(0, nchunks+1):
        _fsname = os.path.join(out_dir_, "chunk_{0}.pkl".format(nc+1))
        if index+chunksize >= list_of_dict.__len__(): chunksize = (list_of_dict.__len__() - index -1)
        with open(_fsname, 'wb') as _fs:
            pickle.dump(list_of_dict[index:index + chunksize], _fs)
            _fs.close()
            print("====> Generated: [{0}]".format(_fsname))
        index += chunksize


def nearest_xy(row, srtree):
    point_ = Point(row.x, row.y)
    points_c = srtree.nearest(point_).centroid
    return points_c.x, points_c.y

def extract_nearest_grid_cell_(row,srtree, x_col_name, y_col_name):
    # dataframe_row,  SRTree,
    row[x_col_name],row[y_col_name]  = nearest_xy(row, srtree)
    return row

def nearest_xy_from_points(x, y, srtree):
    point_ = Point(x, y)
    points_c = srtree.nearest(point_).centroid
    return points_c.x, points_c.y
