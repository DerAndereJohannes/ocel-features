import pandas as pd


def conv_dict_to_pandas(feature_list, obj_dict):
    row_object = list()
    data_x = [None]*len(obj_dict.keys())
    for i, row in enumerate(obj_dict.keys()):
        row_object.append(row)
        data_x[i] = obj_dict[row]

    df = pd.DataFrame(data_x)
    df.transpose()
    df.columns = feature_list

    return df, row_object
