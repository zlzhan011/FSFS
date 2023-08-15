import  pandas as pd
import  re
import  numpy as np
import os
import openpyxl


def category_to_number(data_df, columns_name, columns_value_dict ):

    all_row_value = []
    for index,row in data_df.iterrows():
        row_value = []
        for column in columns_name:
            column_value = row[column]
            if isinstance(column_value,str):
                column_value = column_value.strip()
            if column in columns_value_dict:

                if column_value in columns_value_dict[column]:
                    column_value = columns_value_dict[column][column_value]
                else:
                    column_value = len(list(columns_value_dict[column].keys())) + 1
            else:
                column_value = column_value
            row[column] = column_value
        all_row_value.append(row)

    return pd.DataFrame(all_row_value)


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
    for column in ['age','priors_count','days_b_screening_arrest', 'decile_score', 'c_jail_in', 'c_jail_out']:
        data = discrete_column(data, column)
    return data


def compas_preprocess(data_df):

    category_columns = ['c_charge_degree', 'race', 'age_cat', 'score_text', 'sex', 'priors_count',
              'decile_score', 'c_jail_in', 'c_jail_out']
    filter_condition = data_df['race'].isin(['African-American', 'Caucasian'])
    data_df = data_df[filter_condition]

    columns_value_dict = {}
    for category_column in category_columns:
        if category_column == 'r_days_from_arrest':
            print("------")
        column_list = list(set([item for item in list(data_df[category_column]) if str(item)!='nan']))
        column_dict = {}
        for k in column_list:
            v = column_list.index(k)
            column_dict[k] = v
        columns_value_dict[category_column] = column_dict
    columns_value_dict['race']['Other'] = 1
    columns_value_dict['race']['Native American'] = 1
    columns_value_dict['race']['Hispanic'] = 1
    columns_value_dict['race']['Caucasian'] = 1
    columns_value_dict['race']['Asian'] = 1
    columns_value_dict['race']['African-American'] = 0

    data_df = category_to_number(data_df, category_columns, columns_value_dict)
    data_df = discrete(data_df)
    return  data_df



