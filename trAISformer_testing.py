import torch
import pickle
import models, trainers, datasets, utils
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
import argparse
from p_tqdm import p_map
from shapely.strtree import STRtree
import geopandas as gp
from utils import extract_nearest_grid_cell_, nearest_xy_from_points
replace_xy_with_grid_center = "/home/jayk/wsp_data_tracks/grid_500m_4269_FAOI.shp"
from functools import partial

def define_args():
    parser = argparse.ArgumentParser(description="Testing TrAISFormer",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-dir", "--directory", type=str, default=None, required=True)
    parser.add_argument("-model", "--model", type=str, default=None, required=False)
    parser.add_argument("-cpu", "--cpu", action="store_true")
    args = parser.parse_args()
    return args

def tran_xy(_coords, transformer_to, replace_xy_with_grid_center = None):
    dv_ = _coords.device
    _coords = _coords.detach().cpu().numpy()
    tr_x, tr_y = transformer_to.transform(xx=_coords[:, :, 0], yy=_coords[:, :, 1])
    if replace_xy_with_grid_center is not None:
        for i in range(0, tr_x.shape[0]):
            for j in range(0, tr_x.shape[1]):
                tr_x[i][j], tr_y[i][j] = nearest_xy_from_points( tr_x[i][j], tr_y[i][j])
    grid_x, grid_y = transformer_to.transform(xx=_coords[:, :, 4], yy=_coords[:, :, 5])
    last_grid_x, last_grid_y = transformer_to.transform(xx=_coords[:, :, 6], yy=_coords[:, :, 7])
    gcx_x, gcx_y = transformer_to.transform(xx=_coords[:, :, 8], yy=_coords[:, :, 9])
    t_ = np.stack((tr_x, tr_y, _coords[:, :, 2], _coords[:, :, 3], grid_x, grid_y, last_grid_x, last_grid_y, gcx_x, gcx_y), axis=2)
    t_ = torch.Tensor(t_).to(dv_)
    return t_


def save_plot(pred_errors, cf, model_cofig_dir_):
    plt.figure(figsize=(9, 6), dpi=150)
    v_times = np.arange(len(pred_errors)) / cf.pph
    plt.plot(v_times, pred_errors)

    for nth in range(1, cf.next_hours):
        timestep = cf.pph * nth
        plt.plot(nth, pred_errors[timestep], "o")
        plt.plot([nth, nth], [0, pred_errors[timestep]], "r")
        plt.plot([0, nth], [pred_errors[timestep], pred_errors[timestep]], "r")
        plt.text(nth+0.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

    plt.xlabel("Time (hours)")
    plt.ylabel("Prediction errors (km)")
    plt.xlim([0, 12])
    # plt.ylim([0, 30])
    # plt.ylim([0,pred_errors.max()+0.5])
    plt.savefig(model_cofig_dir_ + "prediction_error.png")

    with open(model_cofig_dir_ + "prediction_error.pkl", 'wb') as pklf_:
        pickle.dump(pred_errors, pklf_)
        pklf_.close()

def export_csv(list_of_x_dfs, list_of_preds_dfs, model_cofig_dir_):
    _x_df_ = pd.concat(list_of_x_dfs, ignore_index=True)
    _preds_df_ = pd.concat(list_of_preds_dfs)
    _x_df_.to_csv(model_cofig_dir_ + "_input_x.csv", index=False)
    _preds_df_.to_csv(model_cofig_dir_ + "_pred_.csv", index=False)


def extract_df_from_pred(rowID, idx_c, preds=None, inputs=None, cf=None, replace_xy_with_grid_center=None, xy_grids_tree=None):
    preds_ = preds[rowID]
    temp_pd = pd.DataFrame(preds_[cf.init_seqlen:, :],
                           columns=['x', 'y', 'speed', 'course', 'cx', 'cy', 'cx_last', 'cy_last'])
    temp_in = pd.DataFrame(inputs[rowID],
                           columns=['x', 'y', 'speed', 'course', 'cx', 'cy', 'cx_last', 'cy_last'])
    temp_pd["T_ID"] = "{0}".format(idx_c)
    temp_in["T_ID"] = "{0}".format(idx_c)

    if replace_xy_with_grid_center is not None:
        temp_pd = temp_pd.apply(
            lambda row: extract_nearest_grid_cell_(row, xy_grids_tree, x_col_name='x', y_col_name='y'),
            axis=1)

    idx_c += 1
    return temp_in, temp_pd

def test_(model, cf, aisdls, replace_xy_with_grid_center = None, device = None):
    transformer_to_4269 = Transformer.from_crs(3857, 4269, always_xy=True)
    if replace_xy_with_grid_center is not None:
        replace_xy_with_grid_center = gp.read_file(replace_xy_with_grid_center)
        xy_grids_tree = STRtree(replace_xy_with_grid_center.geometry)

    if device is not None:
        cf.device = device
        print(cf.device)
    max_seqlen = cf.init_seqlen + cf.pph * cf.next_hours

    l_min_errors, l_mean_errors, l_masks = [], [], []
    pbar = tqdm(enumerate(aisdls["test"]), total=len(aisdls["test"]))

    idx_c = 0
    list_of_input_x = []
    list_of_temp_pd = []
    with torch.no_grad():
        v_ranges = torch.tensor(
            [cf.lat_max - cf.lat_min, cf.lon_max - cf.lon_min, cf.sog_size, cf.cog_max - cf.cog_min, 0, 0, 0, 0, 0, 0]).to(
            cf.device)
        v_roi_min = torch.tensor([cf.lat_min, cf.lon_min, 0, cf.cog_min, 0, 0, 0, 0, 0,0]).to(cf.device)
        v_one_zero = torch.tensor([1, 1, 0, 0, 0, 0, 0, 0,0,0]).to(cf.device)
        iter_ = 0
        for it, (seqs, masks, seqlens, mmsis, time_starts, route_class_weight) in pbar:
            iter_ += 1
            seqs_init = seqs[:, :cf.init_seqlen, :].to(cf.device)
            inputs = seqs[:, :max_seqlen, :].to(cf.device)
            inputs = tran_xy((inputs * v_ranges + v_roi_min), transformer_to_4269)
            input_coords = (inputs * v_one_zero) * torch.pi / 180
            masks = masks[:, :max_seqlen].to(cf.device)
            batchsize = seqs.shape[0]
            error_ens = torch.zeros((batchsize, max_seqlen - cf.init_seqlen, cf.n_samples)).to(cf.device)
            preds = trainers.sample(model,
                                    seqs_init,
                                    max_seqlen - cf.init_seqlen,
                                    temperature=1.0,
                                    sample=True,
                                    sample_mode=cf.sample_mode,
                                    r_vicinity=cf.r_vicinity,
                                    top_k=cf.top_k)
            preds = tran_xy((preds * v_ranges + v_roi_min), transformer_to_4269)
            pred_coords = (preds * v_one_zero) * torch.pi / 180
            d = utils.haversine(input_coords, pred_coords) * masks
            for i_sample in range(cf.n_samples):
                error_ens[:, :, i_sample] = d[:, cf.init_seqlen:]

            rows = preds.shape[0]
            preds = preds.detach().cpu().numpy()
            inputs = inputs.detach().cpu().numpy()



            for rowID in range(0, rows):
                preds_ = preds[rowID]
                temp_pd = pd.DataFrame(preds_[cf.init_seqlen:,:],
                                       columns=['x', 'y', 'speed', 'course', 'cx', 'cy', 'cx_last', 'cy_last','gc_x', 'gc_y'])
                temp_in = pd.DataFrame(inputs[rowID],
                                       columns=['x', 'y', 'speed', 'course', 'cx', 'cy', 'cx_last', 'cy_last','gc_x', 'gc_y'])
                temp_pd["T_ID"] = "{0}".format(idx_c)
                temp_in["T_ID"] = "{0}".format(idx_c)
                list_of_input_x.append(temp_in)

                if replace_xy_with_grid_center is not None:
                    temp_pd = temp_pd.apply(
                        lambda row: extract_nearest_grid_cell_(row, xy_grids_tree, x_col_name='x', y_col_name='y'),
                        axis=1)
                list_of_temp_pd.append(temp_pd)
                idx_c += 1


            if (iter_ == 50):
                export_csv(list_of_input_x, list_of_temp_pd, cf.savedir)
                l_min = [x.values for x in l_min_errors]
                m_masks = torch.cat(l_masks, dim=0)
                min_errors = torch.cat(l_min, dim=0) * m_masks
                pred_errors = min_errors.sum(dim=0) / m_masks.sum(dim=0)
                pred_errors = pred_errors.detach().cpu().numpy()
                save_plot(pred_errors, cf, cf.savedir)


            # Accumulation through batches
            l_min_errors.append(error_ens.min(dim=-1))
            l_mean_errors.append(error_ens.mean(dim=-1))
            l_masks.append(masks[:, cf.init_seqlen:])


    l_min = [x.values for x in l_min_errors]
    m_masks = torch.cat(l_masks, dim=0)
    min_errors = torch.cat(l_min, dim=0) * m_masks
    pred_errors = min_errors.sum(dim=0) / m_masks.sum(dim=0)
    pred_errors = pred_errors.detach().cpu().numpy()

    ## Plot
    # ===============================
    save_plot(pred_errors, cf, cf.savedir)


    # if replace_xy_with_grid_center is not None:
    #     _preds_df_ = _preds_df_.apply(lambda row: extract_nearest_grid_cell_(row, xy_grids_tree, x_col_name='x', y_col_name='y'),axis=1)

    export_csv(list_of_input_x, list_of_temp_pd, cf.savedir)

def do_testing(model, model_cofig_dir_, device = None, replace_xy_with_grid_center = None):
    # transformer_to_3857 = Transformer.from_crs(4269, 3857, always_xy=True)

    aisdls = pickle.load(open(model_cofig_dir_ + "model_config/aisdls.pkl", 'rb'))
    cf = pickle.load(open(model_cofig_dir_ + "model_config/config.pkl", 'rb'))

    test_(model, cf, aisdls, replace_xy_with_grid_center=replace_xy_with_grid_center, device=device)

if __name__ == '__main__':
    args = define_args()


    model_cofig_dir_ = args.directory
    device = None
    # model_cofig_dir_ = "/home/jayk/TrAISformer/results/tracks__2019_h10_d90_s500_t300_rzero_with_recoursing_rc_3857_FAOI_min4h_max6h-pos-pos_vicinity-10-40-blur-True-False-2-1.0-data_size-850-800-30-359-embd_size-256-256-128-128-head-8-8-bs-32-lr-6e-05-seqlen-36-72/"
    model = torch.load(model_cofig_dir_ + "model_config/model.pt")
    if args.cpu:
        device = torch.device('cpu')
        model.to(device)

    if args.model is not None:
        model.load_state_dict(torch.load(model_cofig_dir_+args.model))
    else:
        model.load_state_dict(torch.load(model_cofig_dir_+"model.pt"))

    do_testing(model, model_cofig_dir_, device,
               replace_xy_with_grid_center=replace_xy_with_grid_center
               )

    
