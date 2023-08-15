import os
import pandas as pd
import numpy as np


def convert_sensitive_discrete(array, column_index, percentage=0.7):
    column_value = array[:, column_index]
    sorted_list = sorted(column_value.tolist())
    n = int(len(sorted_list) * percentage)
    digit = sorted_list[n]
    array[:, column_index] = np.where(array[:, column_index] > digit, 1, 0)
    return array









# if __name__ == '__main__':
#     c_root = 'E:\code\GerryFair\dataset\communities'
#     c_root_path = os.path.join(c_root, 'communities.csv')
#     df = pd.read_csv(c_root_path)
#     df = df.drop('Unnamed: 0', axis=1)
#     data_array_train = df.values
#     convert_sensitive_discrete(data_array_train, 2)
