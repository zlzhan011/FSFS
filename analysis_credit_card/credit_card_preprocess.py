import  pandas as pd
import os, re
import numpy as np


def is_digit(s):
    s = str(s).strip()
    digit_find = re.findall(r'[\d.,]',s)
    if len(s) == len(digit_find):
        return 1
    else:
        return 0



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
    for column in ['LIMIT_BAL','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']:
        data = discrete_column(data, column)
    return data

def load_credit_card_v2(file_path):
    diabetes_df = pd.read_excel(file_path)
    diabetes_df = discrete(diabetes_df)

    return diabetes_df




# if __name__ == '__main__':
    # c_root = 'E:\Research\Dataset\Collect_dataset\diabetes_hospital\diabetic_data'
    # diabetes = pd.read_csv(os.path.join(c_root, 'diabetic_data.csv'))
    # load_diabetes(os.path.join(c_root, 'diabetic_data.csv'), os.path.join(c_root, 'diabetic_data.csv'))
    # file_path = r'E:\Research\Dataset\Collect_dataset\credit_card\default of credit card clients.xls'
    # load_credit_card_v2(file_path)

