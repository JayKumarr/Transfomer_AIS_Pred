from .utils import extract_nearest_grid_cell_, nearest_xy,entropy_continuous
from pyproj import CRS, Geod
import numpy as np
import geopandas as gp
import statistics

def process_df_(_df_, data_original_crs,  timestamp=False,
                in_polygon_center = None, xy_grid_tree = None
                , grid_cell_gc = False, grid_cell_indexing_ = None):
    _df_['speed'].mask(_df_['speed'] > 30, 30, inplace=True)


    if _df_.shape[0] < 1:
        return _df_

    if xy_grid_tree is not None:
        _df_ = _df_.apply(lambda row: extract_nearest_grid_cell_(row, xy_grid_tree, x_col_name='x', y_col_name='y'),
                          axis=1)
        _df_['coursing'] = points_to_coursing(_df_.x, _df_.y, data_original_crs)
    if timestamp:
        _df_.sort_values(by=["time"], inplace=True)
        _df_['timestamp'] = _df_['time'].astype(np.int64) / 1000000000



    #  additing nearest grid_cell as feature
    # _df_ = _df_.apply(lambda row: extract_nearest_grid_cell_( row, ins_polygon_center, x_col_name='rcx', y_col_name='cy'), axis=1)
    if in_polygon_center is not None:
        _df_ = _df_.apply(lambda row: extract_nearest_grid_cell_(row, in_polygon_center, "gcx", "gcy"), axis=1)
        if grid_cell_gc:
            _df_['gc_idx'] = _df_.apply(lambda row: grid_cell_indexing_[(row.gcx, row.gcy)], axis=1)



    return _df_

def points_to_coursing(x, y, crs):
    if CRS(crs).is_geographic:
        geod = Geod(ellps="WGS84")
        courses, _, _ = geod.inv(x[:-1], y[:-1], x[1:], y[1:])
    else:
        dx = np.diff(x)
        dy = np.diff(y)
        courses = np.degrees(np.arctan2(dy, dx))
    courses = np.insert(courses, len(courses), courses[-1])
    return courses

def grid_cell_indexing(grid_polygon_shp):
    grid_cell_indexing = {}  # { (x, y) -> index}
    index_iterator = 0
    for polygon in grid_polygon_shp.geometry:
        centr_point = polygon.centroid
        index_iterator+=1
        grid_cell_indexing[(centr_point.x, centr_point.y)] = index_iterator
    return grid_cell_indexing

def get_min_max_grid_relation(grid_cells_dict):
    all_min_max = np.array([])
    for grid_cell_from, dict_ in grid_cells_dict.items():
        for dest_grid, track_ndarray in dict_.items():
            min_stck = track_ndarray.min(axis=0)
            max_stck = track_ndarray.max(axis=0)
            if all_min_max.shape[0] != 0:
                all_min_max = np.vstack([all_min_max, min_stck.reshape(1,6), max_stck.reshape(1,6)])
            else:
                all_min_max = np.stack([min_stck, max_stck], axis=0)

    return all_min_max.min(axis=0), all_min_max.max(axis=0)

def prepare_trackcell_stats(cxs_row_):
    median_ = statistics.median(cxs_row_.coursing)
    entropy_ = entropy_continuous(cxs_row_.coursing, 3)
    # if entropy_ < 0:
    #     continue
    first_x, first_y = cxs_row_.x.iloc[0], cxs_row_.y.iloc[0]
    last_x, last_y = cxs_row_.x.iloc[cxs_row_.x.size - 1], cxs_row_.y.iloc[cxs_row_.y.size - 1]
    c_nparr = np.array([[median_, entropy_, first_x, first_y, last_x, last_y]])
    return c_nparr

def df_normalize(df_, x_min, x_max, y_min, y_max,
                 cog_min,cog_max, sog_max,
                 cx_min, cx_max, cy_min, cy_max,
                 cx_last_min, cx_last_max, cy_last_min, cy_last_max,
                 gcx_min, gcx_max, gcy_min, gcy_max):
    df_.sort_values(by=["time"], inplace=True)
    df_['timestamp'] = df_['time'].astype(np.int64) / 1000000000
    x_range = x_max - x_min
    y_range = y_max - y_min
    cog_range  = cog_max -cog_min
    cx_range = cx_max - cx_min
    cy_range = cy_max - cy_min
    cx_last_range = cx_last_max - cx_last_min
    cy_last_range = cy_last_max - cy_last_min
    gcx_range = gcx_max - gcx_min
    gcy_range = gcy_max - gcy_min


    df_['x'] = (df_['x'] - x_min)/x_range
    df_['y'] = (df_['y'] - y_min)/y_range
    df_['coursing'] = (df_['coursing'] - cog_min)/cog_range
    df_['speed'] = (df_['speed'] - 0)/ (sog_max-0)

    df_['rcx'] = (df_['rcx'] - cx_min)/cx_range
    df_['rcy'] = (df_['rcy'] - cy_min) / cy_range

    df_['cx_last'] = (df_['cx_last'] - cx_last_min) / cx_last_range
    df_['cy_last'] = (df_['cy_last'] - cy_last_min) / cy_last_range

    df_['gcx'] = (df_['gcx'] - gcx_min) / gcx_range
    df_['gcy'] = (df_['gcy'] - gcy_min) / gcy_range

    return df_