
import geopandas as gp
import pandas as pd

def filter_pings_in_poly_from_df(formatted_chunk, polyogn_obj, crs_, x_col = 'X', y_col='Y'):
    """
    :param formatted_chunk: <DataFrame>
    :param polyogn_obj: <Polygon> from Geodataframe
    :param crs_:  <str/int>
    :return: <Dataframe>
    This function filters the pings from formatted_chunk. The x and y must of same crs as polygon.
    """
    gdf_ = gp.GeoDataFrame(formatted_chunk, geometry=gp.points_from_xy(formatted_chunk[x_col], formatted_chunk[y_col]), crs=crs_)
    gdf_ = gdf_[gdf_.geometry.within(polyogn_obj)]
    return pd.DataFrame(gdf_)