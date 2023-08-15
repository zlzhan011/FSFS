import numpy as np
import pandas as pd
import re
import os
import copy


def read_german_file(file_path):
    with open(file_path, 'r', encoding='utf8') as f_read:
        lines = []
        for line in f_read:
            line_split = re.split(" ", line.strip())
            line_split = [int(cell) for cell in line_split if len(cell) > 0]
            line_split = add_gender(line_split)
            lines.append(line_split)
    lines = np.array(lines)
    return lines


def add_gender(line):
    gender_marriage_status = line[6]
    line_y = copy.deepcopy(line[-1])
    if line_y == 2:
        line_y = 0
    if int(gender_marriage_status) in [2,5]:
        gender = 0
    else:
        gender = 1
    line_no_y = line[:-1]

    line_new = line_no_y + [gender, line_y]
    return line_new


def discretize_age(row, mean):
   if row[9] > mean:
      return 1
   else:
       return 0


def german_age_convert(credit_df):
    credit_df[9] = credit_df.apply(lambda row: discretize_age(row, 25), axis=1)
    return credit_df



def discrete_column(data, Column):
    column_to_discretize = data[Column]
    sorted_column = np.sort(column_to_discretize)
    percentiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    bins = np.percentile(sorted_column, percentiles)
    bins = sorted(list(set(bins)))
    bins[0] = bins[0] - 1
    bins[-1] = bins[-1] + 1
    labels = [i for i in range(1, len(bins))]

    data[Column] = pd.cut(column_to_discretize, bins=bins, labels=labels)
    return data


def discrete(data):
    for column in [1,2,3]:
        data = discrete_column(data, column)
    return data

# if __name__ == '__main__':
#     c_root = 'E:\Research\Dataset\Collect_dataset\German'
#     f_path = os.path.join(c_root, 'german.data-numeric')
#     res = read_german_file(f_path)
#     print(res)
