
import json
import os
import numpy as np

def format_column_names(chunk, sup_path=f'{os.path.dirname(__file__)}'):
    """
    The function takes the dataframe and replace the column names in standard format. It uses AIS_Code.json and std_ais_columns.json which are in the same package.
    :param chunk: <dataframe>
    :param sup_path: <str> directory path of json file
    :return: <dataframe>
    """
    #read column name mapper
    file = f'{sup_path}/std_ais_columns.json'
    with open(file, 'r') as f:
        names = json.load(f)
    mapper = {}
    for k in names.keys():
        for v in names[k]:
            mapper[v] = k
            mapper[v.lower()] = k
            mapper[v.upper()] = k
            mapper[v.title()] = k
            mapper[v.replace('_',' ')] = k
            mapper[v.replace('-',' ')] = k
            mapper[v.replace('-','_')] = k
            mapper[v.replace('_','-')] = k
    #make new column names
    new_cols = []
    for col in chunk.columns:
        new_cols.append(mapper.get(col, col))
    #make sure all critical columns have been identified
    need = names.keys()
    unid = [n for n in need if n not in new_cols]
    msg = f'\n\nThe following data could not identified from the raw data columns: ' \
          f'{"  ".join(unid)}. \nThe raw data contains: {"  ".join(chunk.columns)}.' \
          f'\nPlease add their corresponding columns to {file}...\n'
    if len(unid) > 0:
        print(msg)
        print("\nIf all raw data columns have been entered and there are still columns"\
              "not being identified, then they are considered missing...\n\n")
    #set new column names
    chunk.columns = new_cols
    #if ABCD components provided instead of Length/Width
    if all([key in chunk.columns for key in ['A','B','C','D']]):
        chunk['Length'] = chunk['A'] + chunk['B']
        chunk['Width'] = chunk['C'] + chunk['D']
        chunk.drop(columns=['A','B','C','D'], inplace=True)
    elif all([key in chunk.columns for key in ['DimensionA',
                                               'DimensionB',
                                               'DimensionC',
                                               'DimensionD']]):
        chunk['Length'] = chunk['DimensionA'] + chunk['DimensionB']
        chunk['Width'] = chunk['DimensionC'] + chunk['DimensionD']
        chunk.drop(columns=['DimensionA',
                            'DimensionB',
                            'DimensionC',
                            'DimensionD'], inplace=True)
    else:
        pass
    #fix the navstat strings/codes
    if 'Status' in chunk.columns:
        navstat_codes = get_navstat_codes()
        chunk.loc[chunk['Status'].isnull(), 'Status'] = 'default'
        new_status = [navstat_codes[s.lower()] if s.lower() not in navstat_codes.values() else s.lower() for s in chunk['Status']]
        chunk['Status'] = new_status
        chunk['Status'] = chunk['Status'].astype(int)
    else:
        pass
    return chunk

def get_navstat_codes(sup_path=f'{os.path.dirname(__file__)}'):
    """
    returns a dictionary of all navigation status codes from NavigationStatus_Codes.json.
    :param sup_path: <str> directory path of the json
    :return:
    """
    with open(f'{sup_path}/NavigationStatus_Codes.json', 'r') as f:
        codes = json.load(f)
        #incase integers outside of 0-15 are encountered
        newkeys = [int(k) if k.isnumeric() else k.lower() for k in codes.keys()]
        #convert the text nan to a real nan
        newkeys = [k if k != 'nan' else np.nan for k in newkeys]
        return dict(zip(newkeys, codes.values()))
