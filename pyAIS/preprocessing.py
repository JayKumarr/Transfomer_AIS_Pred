
from .tools import format_column_names
from . import utils
import numpy as np
import pandas as pd
import re
from pyproj import CRS
from pyproj import Transformer
from tqdm import tqdm


def clean_aiscodes(df):
    """
    convert the AISCOde of vessel type into standard format/ integer.
    :param df: <dataframe>
    :return: <dataframe>
    """
    codes = df['AISCode']
    _codes = []
    for code in codes:
        if isinstance(code, float) and np.isnan(code):
            code = "[1500] Other"
        match = re.search('\d+', code)
        if match and match.group() != '':
            _codes.append(int(match.group()))
        else:
            _codes.append(code)
    codes=pd.Series(_codes)
    codes = codes.apply(lambda x: 0 if 'reserved' in str(x).lower() else x)
    return codes

def split_pings(chunk, groupby, boundary_poly = None, crs_ = 4269, remove_pings_zero_speed=-1):
    """
    It removes the pings having speed less than remove_ping_speed value from Dataframe.
    Filter the pings inside the defined boundary_poly. Sorts the dataframe by column ‘Time’.
    Return the number of rows with unique MMSI. Each column in each row will have list of values. For example.
    :param chunk: <dataframe>
    :param groupby: <str> column name having MMSI or IMO
    :param boundary_poly: <polygon> area of interest polygon
    :param crs_: <str/int>
    :param remove_pings_zero_speed: <float> will remove the pings having less than this speed
    :return: dataframe
    """
    #format the key required ais names (every supplier slightly different)
    formatted_chunk = format_column_names(chunk)
    if boundary_poly != None:
        formatted_chunk = utils.filter_pings_in_poly_from_df(formatted_chunk, boundary_poly, crs_)
        formatted_chunk.drop('geometry', axis=1, inplace=True)
    if remove_pings_zero_speed > -1 :
        formatted_chunk = formatted_chunk[formatted_chunk["Speed"] > remove_pings_zero_speed]

    # convert all strings to datetime, convert to seconds for easier comparison later on
    formatted_chunk['Time'] = pd.to_datetime(formatted_chunk['Time']).values.astype('datetime64[s]').astype('int')
    # sort by time
    formatted_chunk = formatted_chunk.sort_values(by="Time")

    #group by groupby column, append all entries to giant lists
    chunk_ = formatted_chunk.groupby(groupby)[formatted_chunk.columns].agg(list)

    def r_mmsi(row):
        row["MMSI"] = row.MMSI[0]
        return row
    chunk_ = chunk_.apply(lambda row: r_mmsi(row), axis=1)
    chunk_.columns = chunk_.columns.str.lower()
    return chunk_



def project_to_xy(crs1, crs2, df_, x_col = 'x', y_col = 'y'):
        """
        :param crs1: <str/int> 
        :param crs2: <str/int>
        :param df_:  <dataframe>
        :param x_col: <str>
        :param y_col: <str>
        :return: <dataframe>
        The function transforms the projection of latitude and longitude from crs1 to crs2. 
        """
        crs2 = CRS(crs2)
        crs1 = CRS(crs1)
        if crs1.equals(crs2):
            return df_
        transformer_ = Transformer.from_crs(crs1, crs2, always_xy=True)
        def c_app(row, transformer):
            row[x_col], row[y_col] = transformer.transform(row[x_col], row[y_col])
            return row
        df_ = df_.apply(lambda row: c_app(row, transformer_), axis=1)
        return df_

def resample_to_time(seconds, df_):
    """
    :param seconds: <int>
    :param df_:  <dataframe>
    :return: <dataframe>
    linear interpolation of points (x, y) in dataframe with uniform time span.
    The x and y values are projected in EPSG:3857.
    """
    def resample_t(row):
        t = np.array(row.time)  # to seconds
        tq = np.hstack([np.arange(t[0], t[-1], seconds), t[-1:]])
        row.x = np.interp(tq, t, row.x)
        row.y = np.interp(tq, t, row.y)
        row.coursing = np.interp(tq, t, row.coursing)
        row.heading = np.interp(tq, t, row.heading)
        row.speed = np.interp(tq, t, row.speed)
        row.time = pd.to_datetime(tq, unit='s')  # from seconds
        return row
    df_ = df_.apply(lambda row: resample_t(row), axis=1)
    return df_



