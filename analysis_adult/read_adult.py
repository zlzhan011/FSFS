
import  pandas as pd
import  re
import  numpy as np
def read_file(file_path):
    if '.csv' in file_path:
        data_df = pd.read_csv(file_path)

    return data_df


def is_digit(s):
    s = str(s).strip()
    digit_find = re.findall(r'[\d.,]',s)
    if len(s) == len(digit_find):
        return 1
    else:
        return 0


def get_dict_category_to_number(data_df):
    columns_name = data_df.columns.to_list()
    columns_name_index= {}
    for i in range(len(columns_name)):
        if columns_name[i] == 'class':
            columns_name_index[i] = 'salary'
        elif columns_name[i] == 'hours_per_word':
            columns_name_index[i] = 'hours-per-week'
        elif  columns_name[i] == 'marital':
            columns_name_index[i] = 'marital-status'
        elif '_' in columns_name[i]:
            columns_name_index[i] = re.sub('_','-', columns_name[i])
        else:
            columns_name_index[i] = columns_name[i]
    columns_value_dict = {}

    # text_columns = ['workclass', 'education', ]
    for column in columns_name:
        column_value_distinct = set(data_df[column])
        column_value_dict = {}
        value_index = 0
        every_value_is_digit = 0
        for every_value in column_value_distinct:

            column_value_dict[every_value] = value_index
            value_index = value_index + 1
            every_value_is_digit = every_value_is_digit + is_digit(every_value)
            if every_value_is_digit >= 2:
                break
        if every_value_is_digit == value_index:
            pass
        else:
            columns_value_dict[column] = column_value_dict
    workclass = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked','?']
    education = ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool' ]
    marital = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
    occupation=['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces','?']
    relationship = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']
    race = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
    native_country = ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands','?']
    columns_value_dict['workclass'] = {workclass[i]:i for i  in range(len(workclass))}
    columns_value_dict['education'] = {education[i]: i for i in range(len(education))}
    columns_value_dict['marital'] = {marital[i]: i for i in range(len(marital))}
    columns_value_dict['occupation'] = {occupation[i]: i for i in range(len(occupation))}
    columns_value_dict['relationship'] = {relationship[i]: i for i in range(len(relationship))}
    columns_value_dict['native_country'] = {native_country[i]: i for i in range(len(native_country))}
    columns_value_dict['race'] = {race[i]: i for i in range(len(race))}
    columns_value_dict['sex'] = {'Male': 1, 'Female': 0}
    columns_value_dict['class'] = {'>50K': 1, '<=50K': 0}
    return  columns_name, columns_value_dict, columns_name_index


def category_to_number(data_df, columns_name, columns_value_dict ):

    all_row_value = []
    for index,row in data_df.iterrows():
        row_value = []
        for column in columns_name:
            column_value = row[column]
            if isinstance(column_value,str):
                column_value = column_value.strip()
            if column in columns_value_dict:
                # print("column_value:", column_value)
                # print("column:", column)
                column_value = columns_value_dict[column][column_value]
            else:
                column_value = column_value
            row_value.append(column_value)
        all_row_value.append(row_value)
    # print("columns_value_dict:", columns_value_dict.keys())
    # for k, v in columns_value_dict.items():
    #     print("column_name:", k, "-----", v)
    return np.array(all_row_value)



def load_data(file_path_train, file_path_test):
    data_df_train = read_file(file_path_train)
    file_path_test = read_file(file_path_test)
    data_df_train_test = pd.concat([data_df_train, file_path_test], ignore_index=True)
    columns_name, columns_value_dict, columns_name_index = get_dict_category_to_number(data_df_train_test)
    data_array_train = category_to_number(data_df_train, columns_name, columns_value_dict)
    data_array_test = category_to_number(file_path_test, columns_name, columns_value_dict)
    return data_array_train, data_array_test, columns_name_index


