from shapely.geometry import Point
from scipy.stats import gaussian_kde
import numpy as np
import torch
import statistics
from scipy.spatial import distance

def extract_nearest_grid_cell_(row,srtree, x_col_name, y_col_name):
    # dataframe_row,  SRTree,
    row[x_col_name],row[y_col_name]  = nearest_xy(row, srtree)
    return row

def nearest_xy_from_points(x, y, srtree):
    point_ = Point(x, y)
    points_c = srtree.nearest(point_).centroid
    return points_c.x, points_c.y

def nearest_xy(row, srtree):
    """
    :param row: [Row] from Dataframe
    :param srtree: [STRTree] tree of tuples (grid-cell center x,grid-cell center y)
    :return: (float, float)
    Returns the centroid of nearest cell of a point in row
    """
    point_ = Point(row.x, row.y)
    points_c = srtree.nearest(point_).centroid
    return points_c.x, points_c.y


def entropy_continuous(data, num_bins=10):
    if np.unique(data).size == 1:
        return 0.0

    # Estimate the probability density function using KDE
    kde = gaussian_kde(data)

    # Generate points to evaluate the PDF
    x = np.linspace(min(data), max(data), num_bins)

    # Evaluate the PDF at the generated points
    pdf_values = kde.evaluate(x)

    # Normalize the PDF values to obtain probabilities
    probabilities = pdf_values / np.sum(pdf_values)

    # Calculate the entropy using the probabilities
    entr = -np.sum(probabilities * np.log2(probabilities))

    return entr


def top_k_nearest_idx(att_logits, att_idxs, r_vicinity):
    """Keep only k values nearest the current idx.

    Args:
        att_logits: a Tensor of shape (bachsize, data_size).
        att_idxs: a Tensor of shape (bachsize, 1), indicates
            the current idxs.
        r_vicinity: number of values to be kept.
    """
    device = att_logits.device
    idx_range = torch.arange(att_logits.shape[-1]).to(device).repeat(att_logits.shape[0], 1)
    idx_dists = torch.abs(idx_range - att_idxs)
    out = att_logits.clone()
    out[idx_dists >= r_vicinity / 2] = -float('Inf')
    return out

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


def read_nearest_polys(normalized_nparr, possible_destinations, grel_min, grel_max, top_k):
    """
    :param normalized_nparr:  ndarray
    :param possible_destinations:  dict
    :param grel_min: float
    :param grel_max: float
    :param top_k: int
    :return: list[cell_centriod->dict(distances)   ]
    calculate the distance and probability of points and return list of possible destination grid-cells
    """
    gcc_distance = {}  # gridcell-> nearest_trajectory_distance_
    total_tracks_ = 0
    for gc_center, tracks in possible_destinations.items():
        total_tracks_ += tracks.shape[0]
        normalized_tracks_ = normalized_cell_track(tracks, grel_min, grel_max)
        distances = distance.cdist(normalized_tracks_, normalized_nparr, metric='euclidean')
        gcc_distance[gc_center] = distances.min(axis=0)[0]

    if gcc_distance.__len__() <= top_k:
        return list(gcc_distance.keys()), [1]
    # multiply similarity with cell-track population
    mean_of_distances_ = statistics.mean(gcc_distance.values())
    # plt.scatter([i for i in range(gcc_distance.__len__())], gcc_distance.values(), label='A', marker='o')
    # plt.axhline(y=mean_of_distances_)
    for gc_center, tracks in possible_destinations.items():
        initial_dist = gcc_distance[gc_center]
        if initial_dist < mean_of_distances_:
            trck_count_prob = tracks.shape[0] / total_tracks_
            final_prob = initial_dist * (1 - trck_count_prob)
            gcc_distance[gc_center] = final_prob

    # plt.scatter([i for i in range(gcc_distance.__len__())], gcc_distance.values(), label='B', marker='x')
    first_quantile = np.quantile(list(gcc_distance.values()), [0.25])[0]
    # plt.axhline(y=first_quantile)
    # plt.show()
    tmp_ = sorted(gcc_distance.items(), key=lambda x: x[1], reverse=False)
    list_of_dest_grid_ = []
    list_of_dest_grid_dist_ = []
    for idx in range(tmp_.__len__()):
        grid_cell, dist = tmp_[idx]
        if dist < first_quantile:
            list_of_dest_grid_.append(grid_cell)
            list_of_dest_grid_dist_.append(dist)
        if list_of_dest_grid_.__len__() >= top_k:
            break
    sum_ = sum(list_of_dest_grid_dist_)
    prob_ = [(x_/sum_)  for x_ in list_of_dest_grid_dist_]
    return list_of_dest_grid_, prob_

def normalized_cell_track(track_, grel_min, grel_max):
    normalized__ = (track_ - grel_min) / (grel_max - grel_min)
    return normalized__
