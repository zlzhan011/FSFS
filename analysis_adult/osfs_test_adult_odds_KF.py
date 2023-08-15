import os.path


from learning_module.osfs_and_fast_osfs.osfs_z_mb import osfs_z_mb
from mlp_preprocess import mlp_process_2_feature_select
import numpy as np
from analysis_adult.read_adult import load_data, read_file
from analysis_adult.feature_combination import feature_combine, set_sensitive_att_as_label
from discrimination.calculate_discrimination import calculate_original_discrimination
from learning_module.multi_label.multi_label_test_v2 import MLP, GNB, XGB, LR_V2, LR as MLP_Keras
import copy
import pandas as pd
from fairlearn.metrics import equalized_odds_difference
import time
from analysis_adult.adult_process import write_res
from sklearn.model_selection import KFold




def data_loader(file_path="", file_path_test=""):
    if file_path_test == "" and file_path=="":
        file_path = r'E:\code\HoeffdingTree\data\uci\adult\adult_bak.csv'
        file_path_test = r'E:\code\HoeffdingTree\data\uci\adult\adult_test.csv'
        file_path = '/code/HoeffdingTree/data/uci/adult/adult_bak.csv'
        file_path_test = '/code/HoeffdingTree/data/uci/adult/adult_test.csv'
    data_array, data_array_test, columns_name_index = load_data(file_path, file_path_test)

    data_array_copy = copy.deepcopy(data_array)

    index_array = np.arange(len(data_array))
    data_array = np.column_stack((data_array, index_array))
    SEED = 555
    np.random.seed(SEED)
    np.random.shuffle(data_array)


    random_indices = data_array[:, -1]
    with open('random_indices.txt', 'w', encoding='utf8') as f_write:
        random_indices = random_indices.tolist()
        for r_indices in random_indices:
            f_write.write(str(r_indices) + "\n")

    data_array = data_array[:, :-1]


    data_array_test_copy = copy.deepcopy(data_array_test)
    data_array_train_copy = copy.deepcopy(data_array)

    original_discrimination = calculate_original_discrimination(class_label=data_array_test_copy[:, -1],
                                                                discrimination_column=data_array_test_copy[:, 9])


    return data_array_test, data_array, data_array_test_copy, data_array_train_copy, data_array_copy, columns_name_index


def osfs(data_array_test, data_array, data_array_test_copy):
    testdata = data_array_test
    traindata = data_array
    # testdata = traindata
    class_index = data_array.shape[-1] - 1
    # traindata = traindata[:1000,:]
    selectedFeatures, redundancy_feature, independent_feature, time = osfs_z_mb(traindata, class_index, 0.01, max_k=100)




    selectedFeatures_label = copy.deepcopy(selectedFeatures)


    selectedFeatures = [0, 4, 5, 10, 11]

    y_predict = MLP_Keras(traindata[:, selectedFeatures], traindata[:, class_index].reshape((-1, 1)),
                          testdata[:, selectedFeatures], testdata[:, class_index].reshape((-1, 1)))

    original_discrimination = calculate_original_discrimination(class_label=y_predict,
                                                                discrimination_column=data_array_test_copy[:, 9])


    sensitive_index = 9
    sensitive_label_train_dataset, remove_sensitive_train_dataset = set_sensitive_att_as_label(traindata, class_index,
                                                                                               sensitive_index)
    sensitive_label_test_dataset, remove_sensitive_test_dataset = set_sensitive_att_as_label(testdata, class_index,
                                                                                             sensitive_index)
    traindata = sensitive_label_train_dataset
    testdata = sensitive_label_test_dataset
    class_index = traindata.shape[-1] - 1
    selectedFeatures_sensitive, redundancy_feature_sensitive, independent_feature_sensitive, time = osfs_z_mb(traindata,
                                                                                                              class_index,
                                                                                                              0.01,
                                                                                                              max_k=100)


    if isinstance(selectedFeatures, list):
        pass
    else:
        selectedFeatures = selectedFeatures.tolist()

    for item in selectedFeatures_sensitive:
        if item < sensitive_index:
            if item in selectedFeatures:
                selectedFeatures.remove(item)
        else:
            item_add_1 = item + 1
            if item_add_1 in selectedFeatures:
                selectedFeatures.remove(item_add_1)
    if 9 in selectedFeatures:
        selectedFeatures.remove(9)
    selectedFeatures_no_sensitive = copy.deepcopy(selectedFeatures)

    return selectedFeatures_label, selectedFeatures_no_sensitive



from sklearn.metrics import accuracy_score


def acc(two_label_test, y_pred_test):
    acc_0 = 0.0
    for i in range(two_label_test.shape[-1]):
        print("label i:", i)
        acc_0 = accuracy_score(two_label_test[:, i].astype(int), y_pred_test[:, i])
        # acc_1 = accuracy_score(two_label_test[:, i], y_pred_test[:, i])
        print("acc_O_test", acc_0)
    if two_label_test.shape[-1] == 1:
        return acc_0
    else:
        return ""


def model_train_add_admissible_feature_4(data_array_copy, data_array_test_copy, selectedFeatures, columns_name_index,
                                         contain_s=False, model_type="", write_path="adult_result.xlsx",file_path="", file_path_test="", kfold_i = 0):

    if file_path == "" and file_path_test == "":
        file_path = r'E:\code\HoeffdingTree\data\uci\adult\adult_bak.csv'
        file_path_test = r'E:\code\HoeffdingTree\data\uci\adult\adult_test.csv'
        file_path = '/code/HoeffdingTree/data/uci/adult/adult_bak.csv'
        file_path_test = '/code/HoeffdingTree/data/uci/adult/adult_test.csv'
    data_df_train = read_file(file_path)
    data_df_test = read_file(file_path_test)
    time_start_inner = time.time()
    data_array_train_copy = data_array_copy
    mask = data_array_copy[:, -1] == 1
    filtered_data_array_copy = data_array_copy[mask]
    data_array_copy = np.concatenate((data_array_copy, filtered_data_array_copy, filtered_data_array_copy), axis=0)
    np.random.shuffle(data_array_copy)

    class_index = data_array_train_copy.shape[-1] - 1


    selected_features_name = []
    for feature_index in selectedFeatures:
        selected_features_name.append(columns_name_index[feature_index])

    SEED = 555
    np.random.seed(SEED)
    data_df_train = data_df_train.to_numpy()
    np.random.shuffle(data_df_train)
    # data_df_train = data_df_train[:10000, :]
    data_array_train_copy = pd.DataFrame(data_df_train)
    data_array_test_copy = data_df_test
    columns_name_revise = {"capital_loss": "capital-loss",
                           "hours_per_word": "hours-per-week",
                           "native_country": "native-country",
                           "capital_gain": "capital-gain",
                           "marital": "marital-status",
                           "class": "salary",
                           "education_num": "education-num"}
    data_array_train_copy = data_array_train_copy.rename(columns=columns_name_index)
    data_array_test_copy = data_array_test_copy.rename(columns=columns_name_revise)
    X_train_contain_s, X_test_contain_s, y_train_origin, y_test_origin, X_train_no_s, X_test_no_s, \
        sex_label_for_discrimination_test, sex_label_for_discrimination_train = mlp_process_2_feature_select(
        selected_features_name, data_array_train_copy, data_array_test_copy)

    if 'native-country_ Holand-Netherlands' in X_train_no_s.columns.to_list() and 'native-country_ Holand-Netherlands' not in X_test_no_s.columns.to_list():
        X_train_no_s = X_train_no_s.drop('native-country_ Holand-Netherlands', axis=1)

    data_array = np.c_[np.array(X_train_no_s), np.array(y_train_origin).reshape((-1, 1))]
    data_array_test = np.c_[np.array(X_test_no_s), np.array(y_test_origin).reshape((-1, 1))]
    if contain_s:
        data_array = np.c_[np.array(X_train_contain_s), np.array(y_train_origin).reshape((-1, 1))]
        data_array_test = np.c_[np.array(X_test_contain_s), np.array(y_test_origin).reshape((-1, 1))]

    data_array_train_copy = data_array
    data_array_test_copy = data_array_test
    class_index = data_array.shape[-1] - 1

    print("data_array_train_copy shape:", data_array.shape)
    selectedFeatures = [i for i in range(class_index)]


    y_predict = MLP_Keras(data_array_train_copy[:, selectedFeatures],
                          data_array_train_copy[:, class_index].reshape((-1, 1)),
                          data_array_test_copy[:, selectedFeatures],
                          data_array_test_copy[:, class_index].reshape((-1, 1)))

    acc_0 = acc(data_array_test_copy[:, class_index].reshape((-1, 1)), y_predict)


    original_discrimination = calculate_original_discrimination(class_label=y_predict,
                                                                discrimination_column=sex_label_for_discrimination_test)



    odds_difference = equalized_odds_difference(data_array_test_copy[:, class_index].reshape((-1, 1)),
                                                y_predict,
                                                sensitive_features=sex_label_for_discrimination_test)
    time_end_inner = time.time()
    cost_time = time_end_inner - time_start_inner
    write_path = write_path[:-5] + "_"  + str(kfold_i) + ".xlsx"
    write_res(write_path, model_type, acc_0, odds_difference, cost_time)






if __name__ == '__main__':
    c_root = r'E:\code\HoeffdingTree\data\uci\adult'
    file_path = os.path.join(c_root, 'adult_openml_train.csv')
    file_path_test = os.path.join(c_root, 'adult_openml_test.csv')

    kfold = KFold(n_splits=5, random_state=42, shuffle=True)
    adult_openml_all = pd.read_excel(os.path.join(c_root,'adult_openml.xlsx'))

    # kfold_i = 0
    for kfold_i, (train_indices, test_indices) in enumerate(kfold.split(adult_openml_all)):
        kfold_i = kfold_i + 1
        train_data = adult_openml_all.iloc[train_indices]
        test_data = adult_openml_all.iloc[test_indices]


        train_data.to_csv(file_path, index=False)
        test_data.to_csv(file_path_test, index=False)

        data_array_test, data_array, data_array_test_copy, data_array_train_copy, data_array_copy, columns_name_index = data_loader(file_path=file_path, file_path_test=file_path_test)

        selectedFeatures_label, selectedFeatures_no_sensitive = osfs(data_array_test, data_array, data_array_test_copy)





        model_train_add_admissible_feature_4(data_array_copy, data_array_test_copy, [0, 4, 5, 6, 7, 9, 10, 11, 12],
                                             columns_name_index, contain_s=False, model_type="OSFS",
                                             write_path=os.path.join(c_root, 'adult_result.xlsx'),
                                             file_path=file_path, file_path_test=file_path_test, kfold_i = kfold_i)

        model_train_add_admissible_feature_4(data_array_copy, data_array_test_copy, [0, 4,  9, 10, 11],
                                             columns_name_index, contain_s=False, model_type="FS^2-RI",
                                             write_path=os.path.join(c_root, 'adult_result.xlsx'),
                                             file_path=file_path, file_path_test=file_path_test, kfold_i = kfold_i)

        model_train_add_admissible_feature_4(data_array_copy, data_array_test_copy, [0, 4, 9, 10, 11],
                                             columns_name_index, contain_s=False, model_type="FS^2-AD1",
                                             write_path=os.path.join(c_root, 'adult_result.xlsx'),
                                             file_path=file_path, file_path_test=file_path_test, kfold_i = kfold_i)




        model_train_add_admissible_feature_4(data_array_copy, data_array_test_copy, [0, 4,  9, 10, 11, 13],
                                             columns_name_index, contain_s=False, model_type="FS^2-AD2",
                                             write_path=os.path.join(c_root, 'adult_result.xlsx'),
                                             file_path=file_path, file_path_test=file_path_test, kfold_i = kfold_i)

        model_train_add_admissible_feature_4(data_array_copy, data_array_test_copy, [i for i in range(data_array_copy.shape[-1]-1)],
                                             columns_name_index, contain_s=False, model_type="Remove S",
                                             write_path=os.path.join(c_root, 'adult_result.xlsx'),
                                             file_path=file_path, file_path_test=file_path_test, kfold_i = kfold_i)


        model_train_add_admissible_feature_4(data_array_copy, data_array_test_copy,
                                             [i for i in range(data_array_copy.shape[-1] - 1)],
                                             columns_name_index, contain_s=True, model_type="All Feature",
                                             write_path=os.path.join(c_root, 'adult_result.xlsx'),
                                             file_path=file_path, file_path_test=file_path_test, kfold_i = kfold_i)

