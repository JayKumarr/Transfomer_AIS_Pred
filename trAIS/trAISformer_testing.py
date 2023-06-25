import torch
from tqdm import tqdm
import numpy as np
from .utils import nearest_xy_from_points, extract_nearest_grid_cell_
from . import trainers
import pandas as pd
import pickle

def tran_xy(_coords, transformer_to, replace_xy_with_grid_center = None):
    """
    Convert lat, lon, cx, cy, cx_last, cy_last, gcx, gcy into another projection system.
    if replace_xy_with_grid_center is not none, the original lat and lon will be replaced by the center of nearest polygon.
    :param _coords: <Tensor>
    :param transformer_to:  <pyproj.Trasformer>
    :param replace_xy_with_grid_center: <shapely.strtree.STRTree> A tree containing the polygons (cells).
    :return: <Tensor>
    """
    dv_ = _coords.device
    _coords = _coords.detach().cpu().numpy()
    tr_x, tr_y = transformer_to.transform(xx=_coords[:, :, 0], yy=_coords[:, :, 1])
    if replace_xy_with_grid_center is not None:
        for i in range(0, tr_x.shape[0]):
            for j in range(0, tr_x.shape[1]):
                tr_x[i][j], tr_y[i][j] = nearest_xy_from_points( tr_x[i][j], tr_y[i][j], replace_xy_with_grid_center)
    grid_x, grid_y = transformer_to.transform(xx=_coords[:, :, 4], yy=_coords[:, :, 5])
    last_grid_x, last_grid_y = transformer_to.transform(xx=_coords[:, :, 6], yy=_coords[:, :, 7])
    gcx_x, gcx_y = transformer_to.transform(xx=_coords[:, :, 8], yy=_coords[:, :, 9])
    t_ = np.stack((tr_x, tr_y, _coords[:, :, 2], _coords[:, :, 3], grid_x, grid_y, last_grid_x, last_grid_y, gcx_x, gcx_y), axis=2)
    t_ = torch.Tensor(t_).to(dv_)
    return t_


class TrAIS_Testing:
    def __init__(self, model_path, config_path):
        self.cf = pickle.load(open(config_path, 'rb'))
        self.model = torch.load(model_path)

        self.model.to(self.cf.device)


    def testing(self, aisdls, transformer_to_4269,  xy_grids_tree = None):
        cf = self.cf
        max_seqlen = cf.init_seqlen + cf.pph * cf.next_hours


        idx_c = 0
        list_of_input_x = []
        list_of_temp_pd = []
        with torch.no_grad():
            v_ranges = torch.tensor(
                [cf.lat_max - cf.lat_min, cf.lon_max - cf.lon_min, cf.sog_size, cf.cog_max - cf.cog_min, 0, 0, 0, 0, 0,
                 0]).to(
                cf.device)
            v_roi_min = torch.tensor([cf.lat_min, cf.lon_min, 0, cf.cog_min, 0, 0, 0, 0, 0, 0]).to(cf.device)
            v_one_zero = torch.tensor([1, 1, 0, 0, 0, 0, 0, 0, 0, 0]).to(cf.device)
            iter_ = 0
            for seqs, masks, seqlens, mmsis, time_starts, route_class_weight in aisdls:
                iter_ += 1
                seqs_init = seqs[:, :cf.init_seqlen, :].to(cf.device)
                inputs = seqs[:, :max_seqlen, :].to(cf.device)
                inputs = tran_xy((inputs * v_ranges + v_roi_min), transformer_to_4269)

                batchsize = seqs.shape[0]
                preds = trainers.sample(self.model,
                                        seqs_init,
                                        max_seqlen - cf.init_seqlen,
                                        temperature=1.0,
                                        sample=True,
                                        sample_mode=cf.sample_mode,
                                        r_vicinity=cf.r_vicinity,
                                        top_k=cf.top_k)
                preds = tran_xy((preds * v_ranges + v_roi_min), transformer_to_4269)

                rows = preds.shape[0]
                preds = preds.detach().cpu().numpy()
                inputs = inputs.detach().cpu().numpy()

                for rowID in range(0, rows):
                    preds_ = preds[rowID]
                    temp_pd = pd.DataFrame(preds_[cf.init_seqlen:, :],
                                           columns=['x', 'y', 'speed', 'course', 'cx', 'cy', 'cx_last', 'cy_last',
                                                    'gc_x', 'gc_y'])
                    temp_in = pd.DataFrame(inputs[rowID],
                                           columns=['x', 'y', 'speed', 'course', 'cx', 'cy', 'cx_last', 'cy_last',
                                                    'gc_x', 'gc_y'])
                    temp_pd["T_ID"] = "{0}".format(idx_c)
                    temp_in["T_ID"] = "{0}".format(idx_c)
                    list_of_input_x.append(temp_in)

                    if xy_grids_tree is not None:
                        temp_pd = temp_pd.apply(
                            lambda row: extract_nearest_grid_cell_(row, xy_grids_tree, x_col_name='x', y_col_name='y'),
                            axis=1)
                    list_of_temp_pd.append(temp_pd)
                    idx_c += 1

        # if replace_xy_with_grid_center is not None:
        #     _preds_df_ = _preds_df_.apply(lambda row: extract_nearest_grid_cell_(row, xy_grids_tree, x_col_name='x', y_col_name='y'),axis=1)
        return list_of_temp_pd
        # export_csv(list_of_input_x, list_of_temp_pd, cf.savedir)