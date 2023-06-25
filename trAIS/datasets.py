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



    def __len__(self):
        return len(self.l_data)

    def __getitem__(self, idx):
        """Gets items.
        this function is called when the object of this class is called in 'for' loop
        Returns:
            seq: Tensor of (max_seqlen, [lat,lon,sog,cog]).
            mask: Tensor of (max_seqlen, 1). mask[i] = 0.0 if x[i] is a
            padding.
            seqlen: sequence length.
            mmsi: vessel's MMSI.
            time_start: timestamp of the starting time of the trajectory.
        """
        V = self.l_data[idx]
        m_v = V["traj"][:, [0, 1, 2, 3, 5, 6, 7, 8, 9, 10]]  # lat, lon, sog, cog, cx, cy, cx_last, cy_last, gcx, gcy
        # cx, cy is the center point of route polygon
        #         m_v[m_v==1] = 0.9999
        m_v[m_v > 0.9999] = 0.9999
        seqlen = min(len(m_v), self.max_seqlen)
        seq = np.zeros((self.max_seqlen, 10))
        seq[:seqlen, :] = m_v[:seqlen, :]
        seq = torch.tensor(seq, dtype=torch.float32)

        mask = torch.zeros(self.max_seqlen)
        mask[:seqlen] = 1.

        seqlen = torch.tensor(seqlen, dtype=torch.int)
        mmsi = torch.tensor(V["mmsi"], dtype=torch.int)
        time_start = torch.tensor(int(V["traj"][0, 4]), dtype=torch.int)
        # route_class_weight_ = 1 + self.destination_count_dict[(V["traj"][0, 7], V["traj"][1, 8])]
        route_class_weight_ = 1

        return seq, mask, seqlen, mmsi, time_start, route_class_weight_

