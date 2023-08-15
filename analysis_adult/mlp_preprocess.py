import copy
import os.path

import numpy
import tensorflow as tf
# import tensorflow_addons as tfa
from tensorflow import keras
import numpy as np

np.random.seed(2023)
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from tensorflow import keras
from keras import backend as K
from sklearn.utils import shuffle
from keras.utils import to_categorical
from discrimination.calculate_discrimination import calculate_original_discrimination


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

"""





i: 4 ----column_name: education_num
i: 5 ----column_name: marital
i: 10 ----column_name: capital_gain
i: 11 ----column_name: capital_loss
i: 12 ----column_name: hours_per_word
i: 13 ----column_name: native_country
i: 14 ----column_name: class

"""

def mlp_process_core(selected_features_names, df_train):
    if 'salary' not in selected_features_names:
        selected_features_names.append('salary')
    # 1. Load the data
    header = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
              'relationship',
              'race', 'sex', 'capital-gain',
              'capital-loss', 'hours-per-week', 'native-country', 'salary']
    try:
        file_path = r'E:\code\HoeffdingTree\data\uci\adult\adult_train_test.csv'
        df_temp = pd.read_csv(file_path, index_col=False, skipinitialspace=True)
    except:
        df_temp = pd.read_csv(
            "https://raw.githubusercontent.com/aliakbarbadri/mlp-classifier-adult-dataset/master/adults.csv",
            index_col=False, skipinitialspace=True, header=None, names=header)
    # df_temp = pd.concat([df_train, df_test])
    df_temp = df_train
    df_temp = df_temp[selected_features_names]
    df = df_temp

    # 2. Preprocessing
    # 2.1. Drop rows with NaN value
    # 2.1.1. change ? to np.nan
    df = df.replace('?', np.nan)
    df[pd.isnull(df).any(axis=1)].shape

    # 2.1.2. Drop rows containing nan
    df.dropna(inplace=True)
    """:
    2.2. Deal with categorical columns
    categorical columns

    workclass
    education
    education-num
    marital-status
    occupation
    relationship
    race
    sex
    native-country
    salary
    2.2.1. drop the education-num column because there is education column


    """

    # 2.2.1. drop the education-num column because there is education column
    # if 'education-num' in selected_features_names:
    #     df.drop('education-num', axis=1, inplace=True)

    categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex'
        , 'native-country']

    label_column = ['salary']

    categorical_columns = list(set(categorical_columns) & set(selected_features_names))

    # 2.2.2. convert categoricals to numerical
    def show_unique_values(columns):
        for column in columns:
            uniq = df[column].unique().tolist()
            for i in range(len(uniq)):
                if isinstance(uniq[i],str):
                    uniq[i] = uniq[i].strip()
            # print(column + " has " + str(len(uniq)) + " values" + " : " + str(uniq))

    show_unique_values(categorical_columns)
    show_unique_values(label_column)

    # 2.2.2.1. convert to int
    def convert_to_int(columns):
        for column in columns:
            unique_values = df[column].unique().tolist()
            dic = {}
            for indx, val in enumerate(unique_values):
                dic[val] = indx
            df[column] = df[column].map(dic).astype(int)
            print(column + " done!")

    convert_to_int(label_column)
    show_unique_values(label_column)

    # 2.2.2.2. convert to one-hot (good one)
    def convert_to_onehot(data, columns):
        dummies = pd.get_dummies(data[columns])
        data = data.drop(columns, axis=1)
        data = pd.concat([data, dummies], axis=1)
        return data

    df = convert_to_onehot(df, categorical_columns)

    print(
        """
        2.3. Normalize
        Numerical columns:

        age
        fnlwgt
        capital-gain
        capital-loss
        hours-per-week

        """)

    normalize_columns = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
    normalize_columns = list(set(normalize_columns) & set(selected_features_names))

    def show_values(columns):
        for column in columns:
            max_val = df[column].max()
            min_val = df[column].min()
            mean_val = df[column].mean()
            var_val = df[column].var()
            print(column + ': values=[' + str(min_val) + ',' + str(max_val) + '] , mean=' + str(
                mean_val) + ' , var=' + str(var_val))

    show_values(normalize_columns)

    df = shuffle(df)

    df_1 = df

    def normalize(columns):
        scaler = preprocessing.StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])

    normalize(normalize_columns)
    show_values(normalize_columns)

    # 2.4. Split the data to train, validation and test

    from sklearn.model_selection import train_test_split

    x_data = df.drop('salary', axis=1)
    y_labels = df['salary']
    return x_data, y_labels




def mlp_process_discrete_continuous(X_train, X_test, discrete_column, continuous_column, convert_flag=True):
    if not isinstance(discrete_column,list):
        discrete_column = discrete_column.tolist()
    if not isinstance(continuous_column, list):
        continuous_column = continuous_column.tolist()

    select_features =discrete_column + continuous_column
    if not convert_flag:
        return  X_train[:,select_features], X_test[:,select_features]
    df = np.concatenate((X_train, X_test), axis=0)
    df = pd.DataFrame(df)
    # 1. Preprocessing
    # 1.1. Drop rows with NaN value
    # 1.1.1. change ? to np.nan
    df = df.replace('?', np.nan)
    df[pd.isnull(df).any(axis=1)].shape

    # 1.1.2. Drop rows containing nan
    df.dropna(inplace=True)
    """:
    2.2. Deal with categorical columns
    categorical columns

    workclass
    education
    education-num
    marital-status
    occupation
    relationship
    race
    sex
    native-country
    salary
    2.2.1. drop the education-num column because there is education column


    """


    # categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex'
    #     , 'native-country']

    # label_column = ['salary']

    # categorical_columns = list(set(categorical_columns) & set(selected_features_names))
    categorical_columns = discrete_column
    # 2.2.2. convert categoricals to numerical
    def show_unique_values(columns):
        for column in columns:
            uniq = df[column].unique().tolist()
            for i in range(len(uniq)):
                if isinstance(uniq[i], str):
                    uniq[i] = uniq[i].strip()
            # print(str(column) + " has " + str(len(uniq)) + " values" + " : " + str(uniq))

    show_unique_values(categorical_columns)
    # show_unique_values(label_column)

    # 2.2.2.1. convert to int
    def convert_to_int(columns):
        for column in columns:
            unique_values = df[column].unique().tolist()
            dic = {}
            for indx, val in enumerate(unique_values):
                dic[val] = indx
            df[column] = df[column].map(dic).astype(int)
            print(column + " done!")

    # convert_to_int(label_column)
    # show_unique_values(label_column)

    # 2.2.2.2. convert to one-hot (good one)
    def convert_to_onehot(data, columns):
        dummies = pd.get_dummies(data[columns].astype(str))
        data = data.drop(columns, axis=1)
        data = pd.concat([dummies, data], axis=1)
        return data

    df = convert_to_onehot(df, categorical_columns)

    print(
        """
        2.3. Normalize
        Numerical columns:

        age
        fnlwgt
        capital-gain
        capital-loss
        hours-per-week

        """)

    # normalize_columns = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
    # normalize_columns = list(set(normalize_columns) & set(selected_features_names))
    normalize_columns = continuous_column
    def show_values(columns):
        for column in columns:
            max_val = df[column].max()
            min_val = df[column].min()
            mean_val = df[column].mean()
            var_val = df[column].var()
            print(column + ': values=[' + str(min_val) + ',' + str(max_val) + '] , mean=' + str(
                mean_val) + ' , var=' + str(var_val))

    show_values(normalize_columns)



    def normalize(columns):
        scaler = preprocessing.StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])
    if len(normalize_columns) > 0:
        normalize(normalize_columns)
        show_values(normalize_columns)


    X_train_processed = df.values[0:X_train.shape[0],:]
    X_test_processed = df.values[X_train.shape[0]:, :]
    return X_train_processed, X_test_processed






def mlp_process_2_feature_select(selected_features_names, df_train,df_test):
    train_test = pd.concat([df_train, df_test], ignore_index=True)
    X_train_test, y_train_test = mlp_process_core(selected_features_names, train_test)
    X_train = X_train_test.iloc[0:df_train.shape[0]]
    X_test = X_train_test.iloc[df_train.shape[0]:]

    y_train = y_train_test.iloc[0:df_train.shape[0]]
    y_test = y_train_test.iloc[df_train.shape[0]:]


    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)


    X_train = X_train.rename(columns={"sex_ Male":"sex_Male",
                                                                "sex_ Female":"sex_Female"})
    X_test = X_test.rename(columns={"sex_ Male": "sex_Male",
                                      "sex_ Female": "sex_Female"})
    def get_sex(X_train):
        gender = []
        for index, row in X_train.iterrows():
            sex_Male = row['sex_Male']
            sex_Female = row['sex_Female']
            if sex_Male:
                gender.append(1)
            else:
                gender.append(0)
        return gender

    def combine_two_label(X_test, y_test, X_train, y_train):
        sex_test = np.array(get_sex(X_test)).reshape((-1, 1))
        sex_train = np.array(get_sex(X_train)).reshape((-1, 1))
        y_train = np.array(y_train).reshape((-1, 1))
        y_test = np.array(y_test).reshape((-1, 1))
        y_train_two_label = np.c_[to_categorical(y_train, num_classes=2), to_categorical(sex_train, num_classes=2)]
        y_test_two_label = np.c_[to_categorical(y_test, num_classes=2), to_categorical(sex_test, num_classes=2)]
        return y_train_two_label, y_test_two_label, to_categorical(y_train, num_classes=2), to_categorical(y_test,
                                                                                                           num_classes=2), to_categorical(
            sex_train, num_classes=2), to_categorical(sex_test, num_classes=2)

    sex_label_for_discrimination_test = get_sex(X_test)
    sex_label_for_discrimination_train = get_sex(X_train)
    two_label_flag = True
    drop_sex = True
    X_train_contain_s = copy.deepcopy(X_train)
    # sex_column = X_train_contain_s[['sex_Male', 'sex_Female']]
    # X_train_contain_s.drop('sex_Male', axis=1, inplace=True)
    # X_train_contain_s.drop('sex_Female', axis=1, inplace=True)
    X_test_contain_s = copy.deepcopy(X_test)
    y_train_origin = copy.deepcopy(y_train)
    y_test_origin = copy.deepcopy(y_test)
    if two_label_flag:
        y_train_two_label, y_test_two_label, y_train_label, y_test_label, y_train_label_sex, y_test_label_sex = combine_two_label(
            X_test, y_test, X_train, y_train)

        if drop_sex:
            X_train.drop('sex_Male', axis=1, inplace=True)
            X_train.drop('sex_Female', axis=1, inplace=True)
            X_test.drop('sex_Male', axis=1, inplace=True)
            X_test.drop('sex_Female', axis=1, inplace=True)
            X_train_no_s = copy.deepcopy(X_train)
            X_test_no_s = copy.deepcopy(X_test)

        # y_train = y_train_two_label
        # y_test = y_test_two_label

        y_train = y_train_label
        y_test = y_test_label

        # y_train = y_train_label_sex
        # y_test = y_test_label_sex

    else:
        y_train = np.array(y_train).reshape((-1, 1))
        y_test = np.array(y_test).reshape((-1, 1))

    # new_model.evaluate(X_test, y_test)

    from learning_module.multi_label.multi_label_test_keras import MLP_Keras, evaluate_MLP_Keras
    moder_dir = r'../learning_module/multi_label'
    model_path = os.path.join(moder_dir, 'learning_module/multi_label/model_keras_no_s_s.h5')
    # y_test_predict = MLP_Keras( np.array(X_train), y_train, np.array(X_test),y_test, model_path= model_path, convert_two_label_flag = False)
    # y_test_predict = evaluate_MLP_Keras(model_path, np.array(X_test), y_test)
    #
    # if len(y_test_predict.shape) == 2:
    #     if y_test_predict.shape[-1] == 2:
    #         y_test_predict = y_test_predict[:, 0]
    # original_discrimination = calculate_original_discrimination(class_label=y_test_predict,
    #                                                             discrimination_column=sex_label_for_discrimination_test)
    # print("original_discrimination:", original_discrimination)

    return X_train_contain_s, X_test_contain_s, y_train_origin, y_test_origin, X_train_no_s, X_test_no_s, sex_label_for_discrimination_test, sex_label_for_discrimination_train




