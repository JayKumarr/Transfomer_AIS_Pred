# coding=utf-8
# Copyright 2021, Duong Nguyen
#
# Licensed under the CECILL-C License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.cecill.info
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Customized Pytorch Dataset.
"""

import numpy as np
import os
import pickle

import torch
from torch.utils.data import Dataset, DataLoader

class AISDataset(Dataset):
    """Customized Pytorch dataset.
    """
    def __init__(self, 
                 l_data, 
                 max_seqlen=96,
                 dtype=torch.float32,
                 device=torch.device("cpu")):
        """
        Args
            l_data: list of dictionaries, each element is an AIS trajectory. 
                l_data[idx]["mmsi"]: vessel's MMSI.
                l_data[idx]["traj"]: a matrix whose columns are 
                    [LAT, LON, SOG, COG, TIMESTAMP]
                lat, lon, sog, and cod have been standardized, i.e. range = [0,1).
            max_seqlen: (optional) max sequence length. Default is
        """    
            
        self.max_seqlen = max_seqlen
        self.device = device
        
        self.l_data = l_data

        # This part is to calculate the weight of the destination, because some destination has many tracks
        self.destination_count_dict = {}
        for trck_ in self.l_data:
            dest_x, dest_y=trck_['traj'][0,[7,8]]
            d_xy_ = (dest_x, dest_y)
            if d_xy_ not in self.destination_count_dict:
                self.destination_count_dict[d_xy_] = 0
            self.destination_count_dict[d_xy_] = self.destination_count_dict[d_xy_] + 1

        for d_xy_k in self.destination_count_dict.keys():
            self.destination_count_dict[d_xy_k] = self.destination_count_dict[d_xy_k] / len(self.l_data)

        assert sum(self.destination_count_dict.values()) >= 0.9999 and sum(self.destination_count_dict.values()) <= 1.0000001, "Sum of all values should be 1 because it is probability for each route"

    def __len__(self):
        return len(self.l_data)
        
    def __getitem__(self, idx):
        """Gets items.
        
        Returns:
            seq: Tensor of (max_seqlen, [lat,lon,sog,cog]).
            mask: Tensor of (max_seqlen, 1). mask[i] = 0.0 if x[i] is a
            padding.
            seqlen: sequence length.
            mmsi: vessel's MMSI.
            time_start: timestamp of the starting time of the trajectory.
        """
        V = self.l_data[idx]
        m_v = V["traj"][:,[0,1,2,3,5,6, 7,8,9,10]] # lat, lon, sog, cog, cx, cy, cx_last, cy_last, gcx, gcy
        # cx, cy is the center point of route polygon
#         m_v[m_v==1] = 0.9999
        m_v[m_v>0.9999] = 0.9999
        seqlen = min(len(m_v), self.max_seqlen)
        seq = np.zeros((self.max_seqlen,10))
        seq[:seqlen,:] = m_v[:seqlen,:]
        seq = torch.tensor(seq, dtype=torch.float32)
        
        mask = torch.zeros(self.max_seqlen)
        mask[:seqlen] = 1.
        
        seqlen = torch.tensor(seqlen, dtype=torch.int)
        mmsi =  torch.tensor(V["mmsi"], dtype=torch.int)
        time_start = torch.tensor(int(V["traj"][0,4]), dtype=torch.int)
        route_class_weight_ = 1 + self.destination_count_dict[(V["traj"][0, 7], V["traj"][1, 8])]

        return seq , mask, seqlen, mmsi, time_start, route_class_weight_
    
class AISDataset_grad(Dataset):
    """Customized Pytorch dataset.
    Return the positions and the gradient of the positions.
    """
    def __init__(self, 
                 l_data, 
                 dlat_max=0.04,
                 dlon_max=0.04,
                 max_seqlen=96,
                 dtype=torch.float32,
                 device=torch.device("cpu")):
        """
        Args
            l_data: list of dictionaries, each element is an AIS trajectory. 
                l_data[idx]["mmsi"]: vessel's MMSI.
                l_data[idx]["traj"]: a matrix whose columns are 
                    [LAT, LON, SOG, COG, TIMESTAMP]
                lat, lon, sog, and cod have been standardized, i.e. range = [0,1).
            dlat_max, dlon_max: the maximum value of the gradient of the positions.
                dlat_max = max(lat[idx+1]-lat[idx]) for all idx.
            max_seqlen: (optional) max sequence length. Default is
        """    
            
        self.dlat_max = dlat_max
        self.dlon_max = dlon_max
        self.dpos_max = np.array([dlat_max, dlon_max])
        self.max_seqlen = max_seqlen
        self.device = device
        
        self.l_data = l_data 

    def __len__(self):
        return len(self.l_data)
        
    def __getitem__(self, idx):
        """Gets items.
        
        Returns:
            seq: Tensor of (max_seqlen, [lat,lon,sog,cog]).
            mask: Tensor of (max_seqlen, 1). mask[i] = 0.0 if x[i] is a
            padding.
            seqlen: sequence length.
            mmsi: vessel's MMSI.
            time_start: timestamp of the starting time of the trajectory.
        """
        V = self.l_data[idx]
        m_v = V["traj"][:,[0,1,2,3,5,6, 7,8,9,10]] # lat, lon, sog, cog, cx, cy, cx_last, cy_last
        m_v[m_v==1] = 0.9999
        seqlen = min(len(m_v), self.max_seqlen)
        seq = np.zeros((self.max_seqlen,[0,1,2,3,5,6, 7, 8,9,10]))
        # lat and lon
        seq[:seqlen,:2] = m_v[:seqlen,:2] 
        # dlat and dlon
        dpos = (m_v[1:,:2]-m_v[:-1,:2]+self.dpos_max )/(2*self.dpos_max )
        dpos = np.concatenate((dpos[:1,:],dpos),axis=0)
        dpos[dpos>=1] = 0.9999
        dpos[dpos<=0] = 0.0
        seq[:seqlen,2:] = dpos[:seqlen,:2] 
        
        # convert to Tensor
        seq = torch.tensor(seq, dtype=torch.float32)
        
        mask = torch.zeros(self.max_seqlen)
        mask[:seqlen] = 1.
        
        seqlen = torch.tensor(seqlen, dtype=torch.int)
        mmsi =  torch.tensor(V["mmsi"], dtype=torch.int)
        time_start = torch.tensor(V["traj"][0,10], dtype=torch.int) # check 6-> 8
        
        return seq , mask, seqlen, mmsi, time_start