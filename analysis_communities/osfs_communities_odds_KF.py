
from learning_module.osfs_and_fast_osfs.osfs_z_mb import osfs_z_mb
from statistical_comparsion.knnclassify.knn import knnclassify_calculate_v2
import numpy as np

from discrimination.calculate_discrimination import calculate_original_discrimination

from learning_module.multi_label.multi_label_test_keras import MLP_Keras_V2

import copy, os
import pandas as pd
from analysis_adult.mlp_preprocess import mlp_process_discrete_continuous
from openpyxl import Workbook
from analysis_german.model import modeling
from fairlearn.metrics import equalized_odds_difference
from analysis_communities.comunities_preprocess import convert_sensitive_discrete

import warnings
warnings.filterwarnings("ignore")
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.model_selection import KFold


def revise_p_mb(p_mb, p_index):
    for i in range(len(p_mb)):
        if p_mb[i] < p_index:
            pass
        else:
            p_mb[i] = p_mb[i] + 1
    return p_mb


def find_admissible_features(y_mb, y_redundancy, p_mb, p_redundancy, p_independent, p_index, sheet, cell_number):
    p_mb = revise_p_mb(p_mb, p_index)
    p_redundancy = revise_p_mb(p_redundancy, p_index)
    p_independent = revise_p_mb(p_independent, p_index)
    sheet['D' + str(cell_number + 21)] = str(p_mb)
    sheet['D' + str(cell_number + 22)] = str(p_redundancy)
    sheet['D' + str(cell_number + 23)] = str(p_independent)
    y_mb = set(y_mb)
    p_mb = set(p_mb)
    intersection = list(y_mb.intersection(p_mb))
    if len(intersection) == 0:
        return list(y_mb), list(y_mb), sheet, list(y_mb.difference(p_mb))
    else:
        y_mb_difference = list(y_mb.difference(p_mb))
        y_redundancy = set(y_redundancy)
        p_independent = set(p_independent)
        admissible_i = list(y_redundancy.intersection(p_independent))

        sheet['E' + str(cell_number + 23)] = str(admissible_i)

        admissible_r = []
        for item in list(y_redundancy):
            if item not in list(p_mb):
                admissible_r.append(item)
        sheet['F' + str(cell_number + 22)] = str(admissible_r)
        return list(y_mb_difference) + list(admissible_i), list(y_mb_difference) + list(admissible_r), sheet, y_mb_difference


def process_y(df, sheet, X_train, X_test, y_train, y_test):
    sheet['G27'] = str([i for i in range(X_train.shape[-1])])


    X_train_processed, X_test_processed = mlp_process_discrete_continuous(X_train, X_test,
                                                                          [i for i in range(X_train.shape[-1])], [],
                                                                          convert_flag=convert_flag)


    y_test_predict_all_feature_knn, acc_test_all_feature_knn = knnclassify_calculate_v2(
        [index for index in range(X_train_processed.shape[-1] - 1)],
        np.concatenate((X_train_processed, y_train), axis=1),
        np.concatenate((X_test_processed, y_test), axis=1), n_neighbors=5)
    sheet['G35'] = str(acc_test_all_feature_knn)

    modeling_res = modeling(X_train, y_train, X_test, y_test, write_flag='all_features', c_root=c_root)
    y_test_predict_all_feature = modeling_res['MLP']['y_pre']
    y_test_predict_all_feature_lgbm = modeling_res['LGBM']['y_pre']
    sheet['G29'] = str(modeling_res['MLP']['accuracy'])
    sheet['G32'] = str(modeling_res['LGBM']['accuracy'])
    traindata = np.concatenate((X_train, y_train), axis=1)
    class_index = traindata.shape[-1] - 1

    selectedFeatures, redundancy_feature, independent_feature, time = osfs_z_mb(traindata, class_index, 0.01, max_k=100)

    sheet['C21'] = str(selectedFeatures)
    sheet['C22'] = str(redundancy_feature)
    sheet['C23'] = str(independent_feature)

    sheet['A2'] = 'traindata.shape'
    sheet['B2'] = str(traindata.shape)

    one_res['traindata_shape'] = traindata.shape


    sheet['C27'] = str(selectedFeatures)
    selectedFeatures_y_label = copy.deepcopy(selectedFeatures)

    X_train_processed, X_test_processed = mlp_process_discrete_continuous(X_train, X_test,
                                                                          selectedFeatures, [],
                                                                          convert_flag=convert_flag)



    y_test_predict_y_mb_z_knn, acc_test_y_mb_z_knn = knnclassify_calculate_v2(selectedFeatures, np.concatenate(
        (X_train_processed, y_train), axis=1), np.concatenate((X_test_processed, y_test), axis=1), n_neighbors=5)
    sheet['C35'] = str(acc_test_y_mb_z_knn)
    modeling_res_y_mb = modeling(X_train[:, selectedFeatures], y_train, X_test[:, selectedFeatures], y_test,
                                 write_flag='y_mb', c_root=c_root)
    y_test_predict_y_mb_z = modeling_res_y_mb['MLP']['y_pre']
    y_test_predict_y_mb_z_lgbm = modeling_res_y_mb['LGBM']['y_pre']
    sheet['C29'] = str(modeling_res_y_mb['MLP']['accuracy'])
    sheet['C32'] = str(modeling_res_y_mb['LGBM']['accuracy'])

    return traindata, X_train, X_test, y_train, y_test, sheet, y_test_predict_all_feature, \
           y_test_predict_all_feature_lgbm, y_test_predict_all_feature_knn, selectedFeatures_y_label, \
           redundancy_feature, y_test_predict_y_mb_z,y_test_predict_y_mb_z_lgbm,y_test_predict_y_mb_z_knn, class_index


if __name__ == '__main__':
    c_root = 'E:\code\GerryFair\dataset\communities'
    c_root_path = os.path.join(c_root, 'communities.csv')
    df = pd.read_csv(c_root_path)
    df = df.drop('Unnamed: 0', axis=1)
    df = df.sample(frac=1, random_state=42)
    data_array_train = df.values

    convert_flag = True
    save_res = []
    one_res = {}
    np.random.shuffle(data_array_train)
    df = pd.DataFrame(data_array_train)

    df = df.values
    np.random.shuffle(df)

    sensitive_index_list = [2]

    for si in range(len(sensitive_index_list)):
        sensitive_index = sensitive_index_list[si]
        df = convert_sensitive_discrete(df, sensitive_index)
        kfold = KFold(n_splits=5, random_state=42, shuffle=True)
        for kfold_i, (train_indices, test_indices) in enumerate(kfold.split(df)):
            kfold_i = kfold_i + 1

            workbook = Workbook()
            sheet = workbook.active
            sheet['A1'] = 'all data shape'
            sheet['B1'] = str(df.shape)



            train_data = df[train_indices]
            test_data = df[test_indices]

            X_train, X_test, y_train, y_test = train_data[:, :-1], test_data[:, :-1], train_data[:, -1], test_data[:,
                                                                                                         -1]
            y_train = y_train.reshape((-1, 1))
            y_test = y_test.reshape((-1, 1))



            traindata, X_train, X_test, y_train, y_test, sheet, y_test_predict_all_feature, \
            y_test_predict_all_feature_lgbm, y_test_predict_all_feature_knn, selectedFeatures_y_label, \
            redundancy_feature, y_test_predict_y_mb_z, y_test_predict_y_mb_z_lgbm, y_test_predict_y_mb_z_knn, class_index \
                = process_y(df, sheet, X_train, X_test, y_train, y_test)
            selectedFeatures_y_label = selectedFeatures_y_label.tolist()
            if sensitive_index in selectedFeatures_y_label:
                selectedFeatures_y_label.remove(sensitive_index)





            # remove sensitive feature
            feature_remove_s = [i for i in range(X_train.shape[-1]) if i != sensitive_index]
            modeling_res_y_mb_rs = modeling(X_train[:, feature_remove_s], y_train, X_test[:, feature_remove_s],
                                                   y_test,
                                                   write_flag='y_mb', c_root=c_root)
            y_test_predict_y_mb_z_rs = modeling_res_y_mb_rs['MLP']['y_pre']
            y_test_predict_y_mb_z_lgbm_rs = modeling_res_y_mb_rs['LGBM']['y_pre']
            rs_mlp_acc = str(modeling_res_y_mb_rs['MLP']['accuracy'])
            rs_lgbm_acc = str(modeling_res_y_mb_rs['LGBM']['accuracy'])

            original_discrimination_mlp_rs = calculate_original_discrimination(class_label=y_test_predict_y_mb_z_rs,
                                                                               discrimination_column=X_test[:,
                                                                                                     sensitive_index])
            odds_difference_mlp_rs = equalized_odds_difference(y_test,
                                                               y_test_predict_y_mb_z_rs,
                                                               sensitive_features=X_test[:, sensitive_index])



            original_discrimination_xgb_rs = calculate_original_discrimination(
                class_label=y_test_predict_y_mb_z_lgbm_rs,
                discrimination_column=X_test[:,
                                      sensitive_index])
            odds_difference_rs = equalized_odds_difference(y_test,
                                                           y_test_predict_y_mb_z_lgbm_rs,
                                                           sensitive_features=X_test[:, sensitive_index])





            cell_number = si * 21

            sheet['H' + str(cell_number + 26)] = 'Remove S'
            sheet['H' + str(cell_number + 27)] = str(feature_remove_s)
            sheet['H' + str(cell_number + 28)] = str(odds_difference_mlp_rs)
            sheet['H' + str(cell_number + 29)] = str(rs_mlp_acc)





            sheet['C' + str(cell_number + 20)] = 'label mb'
            sheet['D' + str(cell_number + 20)] = 'sensitive mb'
            sheet['E' + str(cell_number + 20)] = 'admissible features'
            sheet['F' + str(cell_number + 20)] = 'admissible features 2'
            sheet['B' + str(cell_number + 21)] = 'selectedFeatures'
            sheet['B' + str(cell_number + 22)] = 'redundancy_feature'
            sheet['B' + str(cell_number + 23)] = 'independent_feature'
            sheet['C' + str(cell_number + 26)] = 'label mb'
            sheet['E' + str(cell_number + 26)] = 'add admissible features'
            sheet['F' + str(cell_number + 26)] = 'add admissible features 2'
            sheet['G' + str(cell_number + 26)] = 'select all features'
            sheet['B' + str(cell_number + 28)] = 'discrimination score'
            sheet['B' + str(cell_number + 29)] = 'acc'
            sheet['B' + str(cell_number + 31)] = 'discrimination score'
            sheet['B' + str(cell_number + 32)] = 'acc'
            sheet['B' + str(cell_number + 34)] = 'discrimination score'
            sheet['B' + str(cell_number + 35)] = 'acc'
            sheet['A' + str(cell_number + 28)] = 'MLP'
            sheet['A' + str(cell_number + 29)] = 'MLP'
            sheet['A' + str(cell_number + 31)] = 'LGBM'
            sheet['A' + str(cell_number + 32)] = 'LGBM'
            sheet['A' + str(cell_number + 34)] = 'KNN'
            sheet['A' + str(cell_number + 35)] = 'KNN'
            sheet['C' + str(cell_number + 21)] = sheet['C21'].value
            sheet['C' + str(cell_number + 22)] = sheet['C22'].value
            sheet['C' + str(cell_number + 23)] = sheet['C23'].value
            sheet['C' + str(cell_number + 29)] = sheet['C29'].value
            sheet['C' + str(cell_number + 27)] = sheet['C27'].value
            sheet['C' + str(cell_number + 32)] = sheet['C32'].value
            sheet['C' + str(cell_number + 35)] = sheet['C35'].value
            sheet['G' + str(cell_number + 29)] = sheet['G29'].value
            sheet['G' + str(cell_number + 27)] = sheet['G27'].value
            sheet['G' + str(cell_number + 32)] = sheet['G32'].value
            sheet['G' + str(cell_number + 35)] = sheet['G35'].value
            one_res_s = copy.deepcopy(one_res)
            one_res_s['sensitive_index'] = sensitive_index
            sheet['A' + str(cell_number + 17)] = 'sensitive_index'
            sheet['A' + str(cell_number + 18)] = str(sensitive_index)

            original_discrimination = calculate_original_discrimination(class_label=df[:, -1],
                                                                        discrimination_column=df[:, sensitive_index])
            odds_difference = equalized_odds_difference(df[:, -1],
                                                        df[:, -1],
                                                        sensitive_features=df[:, sensitive_index])
            original_discrimination = odds_difference
            sheet['B' + str(cell_number + 17)] = 'original_discrimination all data'
            sheet['B' + str(cell_number + 18)] = str(original_discrimination)
            print("\n\n\n")

            original_discrimination = calculate_original_discrimination(class_label=y_test_predict_all_feature,
                                                                        discrimination_column=X_test[:, sensitive_index])
            odds_difference = equalized_odds_difference(y_test,
                                                        y_test_predict_all_feature,
                                                        sensitive_features=X_test[:, sensitive_index])
            original_discrimination = odds_difference

            sheet['G' + str(cell_number + 28)] = str(original_discrimination)
            original_discrimination_xgb = calculate_original_discrimination(class_label=y_test_predict_all_feature_lgbm,
                                                                            discrimination_column=X_test[:,
                                                                                                  sensitive_index])
            odds_difference = equalized_odds_difference(y_test,
                                                        y_test_predict_all_feature_lgbm,
                                                        sensitive_features=X_test[:, sensitive_index])
            original_discrimination_xgb = odds_difference

            sheet['G' + str(cell_number + 31)] = str(original_discrimination_xgb)
            original_discrimination_knn = calculate_original_discrimination(class_label=y_test_predict_all_feature_knn,
                                                                            discrimination_column=X_test[:,
                                                                                                  sensitive_index])
            odds_difference = equalized_odds_difference(y_test,
                                                        y_test_predict_all_feature_knn,
                                                        sensitive_features=X_test[:, sensitive_index])
            original_discrimination_knn = odds_difference

            sheet['G' + str(cell_number + 34)] = str(original_discrimination_knn)


            distance = class_index - sensitive_index
            if distance == 1:
                traindata_s = traindata[:, :-1]
            else:
                traindata_s = np.concatenate((traindata[:, :sensitive_index], traindata[:, sensitive_index + 1:-1]), axis=1)
                traindata_s = np.concatenate((traindata_s, traindata[:, sensitive_index].reshape((-1, 1))), axis=1)


            sensitive_index_s = traindata_s.shape[-1] - 1
            selectedFeatures_s, redundancy_feature_s, independent_feature_s, time_s = osfs_z_mb(traindata_s,
                                                                                                sensitive_index_s, 0.01,
                                                                                                max_k=100)


            admissible_i, admissible_r, sheet, y_mb_difference = find_admissible_features(selectedFeatures_y_label, redundancy_feature,
                                                                         selectedFeatures_s, redundancy_feature_s,
                                                                         independent_feature_s, sensitive_index, sheet,
                                                                         cell_number)

            sheet['D' + str(cell_number + 26)] = 'remove intersection'
            sheet['D' + str(cell_number + 27)] = str(y_mb_difference)

            selectedFeatures = y_mb_difference
            modeling_res_y_mb_remove_i = modeling(X_train[:, selectedFeatures], y_train, X_test[:, selectedFeatures],
                                                  y_test,
                                                  write_flag='y_mb', c_root=c_root)
            y_test_predict = modeling_res_y_mb_remove_i['MLP']['y_pre']
            y_test_predict_xgb = modeling_res_y_mb_remove_i['LGBM']['y_pre']
            acc_test_admissible_i = str(modeling_res_y_mb_remove_i['MLP']['accuracy'])
            acc_test_admissible_i_xgb = str(modeling_res_y_mb_remove_i['LGBM']['accuracy'])


            original_discrimination = calculate_original_discrimination(class_label=y_test_predict,
                                                                        discrimination_column=X_test[:,
                                                                                              sensitive_index])
            odds_difference = equalized_odds_difference(y_test,
                                                        y_test_predict,
                                                        sensitive_features=X_test[:, sensitive_index])
            original_discrimination = odds_difference

            sheet['D' + str(cell_number + 28)] = str(original_discrimination)
            sheet['D' + str(cell_number + 29)] = str(acc_test_admissible_i)

            original_discrimination = calculate_original_discrimination(class_label=y_test_predict_xgb,
                                                                        discrimination_column=X_test[:,
                                                                                              sensitive_index])
            odds_difference = equalized_odds_difference(y_test,
                                                        y_test_predict_xgb,
                                                        sensitive_features=X_test[:, sensitive_index])
            original_discrimination = odds_difference

            sheet['D' + str(cell_number + 31)] = str(original_discrimination)
            sheet['D' + str(cell_number + 32)] = str(acc_test_admissible_i)






            original_discrimination = calculate_original_discrimination(class_label=y_train,
                                                                        discrimination_column=X_train[:, sensitive_index])
            odds_difference = equalized_odds_difference(y_train,
                                                        y_train,
                                                        sensitive_features=X_train[:, sensitive_index])
            original_discrimination = odds_difference
            sheet['C' + str(cell_number + 17)] = 'original_discrimination train data'
            sheet['C' + str(cell_number + 18)] = str(original_discrimination)

            original_discrimination = calculate_original_discrimination(class_label=y_test,
                                                                        discrimination_column=X_test[:, sensitive_index])
            print("only test dataset, not feature select, discrimination :", original_discrimination)
            odds_difference = equalized_odds_difference(y_test,
                                                        y_test,
                                                        sensitive_features=X_test[:, sensitive_index])
            original_discrimination = odds_difference
            sheet['D' + str(cell_number + 17)] = 'original_discrimination test data'
            sheet['D' + str(cell_number + 18)] = str(original_discrimination)

            original_discrimination = calculate_original_discrimination(class_label=y_test_predict_y_mb_z,
                                                                        discrimination_column=X_test[:, sensitive_index])
            odds_difference = equalized_odds_difference(y_test,
                                                        y_test_predict_y_mb_z,
                                                        sensitive_features=X_test[:, sensitive_index])
            original_discrimination = odds_difference
            sheet['C' + str(cell_number + 28)] = str(original_discrimination)

            original_discrimination_xgb = calculate_original_discrimination(class_label=y_test_predict_y_mb_z_lgbm,
                                                                            discrimination_column=X_test[:,
                                                                                                  sensitive_index])
            odds_difference = equalized_odds_difference(y_test,
                                                        y_test_predict_y_mb_z_lgbm,
                                                        sensitive_features=X_test[:, sensitive_index])
            original_discrimination_xgb = odds_difference

            sheet['C' + str(cell_number + 31)] = str(original_discrimination_xgb)

            original_discrimination_knn = calculate_original_discrimination(class_label=y_test_predict_y_mb_z_knn,
                                                                            discrimination_column=X_test[:,
                                                                                                  sensitive_index])
            odds_difference = equalized_odds_difference(y_test,
                                                        y_test_predict_y_mb_z_knn,
                                                        sensitive_features=X_test[:, sensitive_index])
            original_discrimination_knn = odds_difference
            sheet['C' + str(cell_number + 34)] = str(original_discrimination_knn)

            print("\n\n\n")

            # # add admissible features
            print("&&&&&&&&&add admissible features&&&&&&&&&&&&&")
            if admissible_i == admissible_r:
                one_res_s['admissible_i equal admissible_r'] = admissible_i
                selectedFeatures = admissible_i
                y_test_predict, acc_test_i_equal_r = MLP_Keras_V2(X_train[:, selectedFeatures], y_train,
                                                                  X_test[:, selectedFeatures], y_test)
                one_res_s['acc_test_i_equal_r'] = acc_test_i_equal_r
                original_discrimination = calculate_original_discrimination(class_label=y_test_predict,
                                                                            discrimination_column=X_test[:,
                                                                                                  sensitive_index])

                print("admissible_i == admissible_r after add admissible features discrimination :",
                      original_discrimination)
                one_res_s[
                    'admissible_i equal admissible_r after add admissible features discrimination'] = original_discrimination
            else:
                print("!=!=!=!=!=!=!=!=!=!=admissible_i is equla to admissible_r")
                print("admissible_i:", admissible_i)
                one_res_s['i not equal r admissible_i'] = admissible_i
                selectedFeatures = admissible_i
                sheet['E' + str(cell_number + 27)] = str(selectedFeatures)

                X_train_processed, X_test_processed = mlp_process_discrete_continuous(X_train, X_test,
                                                                                      selectedFeatures, [],
                                                                                      convert_flag=convert_flag)

                modeling_res_y_mb_add_a = modeling(X_train[:, selectedFeatures], y_train, X_test[:, selectedFeatures],
                                                   y_test,
                                                   write_flag='y_mb', c_root=c_root)
                y_test_predict = modeling_res_y_mb_add_a['MLP']['y_pre']
                y_test_predict_xgb = modeling_res_y_mb_add_a['LGBM']['y_pre']
                acc_test_admissible_i = str(modeling_res_y_mb_add_a['MLP']['accuracy'])
                acc_test_admissible_i_xgb = str(modeling_res_y_mb_add_a['LGBM']['accuracy'])


                original_discrimination = calculate_original_discrimination(class_label=y_test_predict,
                                                                            discrimination_column=X_test[:,
                                                                                                  sensitive_index])
                odds_difference = equalized_odds_difference(y_test,
                                                            y_test_predict,
                                                            sensitive_features=X_test[:, sensitive_index])
                original_discrimination = odds_difference

                sheet['E' + str(cell_number + 28)] = str(original_discrimination)
                sheet['E' + str(cell_number + 29)] = str(acc_test_admissible_i)


                original_discrimination_xgb = calculate_original_discrimination(class_label=y_test_predict_xgb,
                                                                                discrimination_column=X_test[:,
                                                                                                      sensitive_index])
                odds_difference = equalized_odds_difference(y_test,
                                                            y_test_predict_xgb,
                                                            sensitive_features=X_test[:, sensitive_index])
                original_discrimination_xgb = odds_difference

                sheet['E' + str(cell_number + 31)] = str(original_discrimination_xgb)
                sheet['E' + str(cell_number + 32)] = str(acc_test_admissible_i_xgb)

                y_test_predict_knn, acc_test_knn = knnclassify_calculate_v2(selectedFeatures,
                                                                            np.concatenate((X_train_processed, y_train),
                                                                                           axis=1),
                                                                            np.concatenate((X_test_processed, y_test),
                                                                                           axis=1),
                                                                            n_neighbors=5)
                original_discrimination_knn = calculate_original_discrimination(class_label=y_test_predict_knn,
                                                                                discrimination_column=X_test[:,
                                                                                                      sensitive_index])
                odds_difference = equalized_odds_difference(y_test,
                                                            y_test_predict_knn,
                                                            sensitive_features=X_test[:, sensitive_index])
                original_discrimination_knn = odds_difference
                sheet['E' + str(cell_number + 34)] = str(original_discrimination_knn)
                sheet['E' + str(cell_number + 35)] = str(acc_test_knn)

                print("admissible_i:", admissible_r)
                one_res_s['i not equal r admissible_r'] = admissible_r
                selectedFeatures = admissible_r
                if sensitive_index in selectedFeatures:
                    selectedFeatures.remove(sensitive_index)
                sheet['F' + str(cell_number + 27)] = str(selectedFeatures)


                modeling_res_y_mb_add_a_r = modeling(X_train[:, selectedFeatures], y_train, X_test[:, selectedFeatures],
                                                     y_test,
                                                     write_flag='y_mb', c_root=c_root)

                y_test_predict = modeling_res_y_mb_add_a_r['MLP']['y_pre']
                y_test_predict_xgb = modeling_res_y_mb_add_a_r['LGBM']['y_pre']
                sheet['F' + str(cell_number + 29)] = str(modeling_res_y_mb_add_a_r['MLP']['accuracy'])
                sheet['F' + str(cell_number + 32)] = str(modeling_res_y_mb_add_a_r['LGBM']['accuracy'])
                original_discrimination = calculate_original_discrimination(class_label=y_test_predict,
                                                                            discrimination_column=X_test[:,
                                                                                                  sensitive_index])
                odds_difference = equalized_odds_difference(y_test,
                                                            y_test_predict,
                                                            sensitive_features=X_test[:, sensitive_index])
                original_discrimination = odds_difference


                sheet['F' + str(cell_number + 28)] = str(original_discrimination)

                original_discrimination_xgb = calculate_original_discrimination(class_label=y_test_predict_xgb,
                                                                                discrimination_column=X_test[:,
                                                                                                      sensitive_index])
                odds_difference = equalized_odds_difference(y_test,
                                                            y_test_predict_xgb,
                                                            sensitive_features=X_test[:, sensitive_index])
                original_discrimination_xgb = odds_difference
                sheet['F' + str(cell_number + 31)] = str(original_discrimination_xgb)


                y_test_predict_knn, acc_test_knn = knnclassify_calculate_v2(selectedFeatures,
                                                                            np.concatenate((X_train_processed, y_train),
                                                                                           axis=1),
                                                                            np.concatenate((X_test_processed, y_test),
                                                                                           axis=1),
                                                                            n_neighbors=5)
                original_discrimination_knn = calculate_original_discrimination(class_label=y_test_predict_knn,
                                                                                discrimination_column=X_test[:,
                                                                                                      sensitive_index])
                odds_difference = equalized_odds_difference(y_test,
                                                            y_test_predict_knn,
                                                            sensitive_features=X_test[:, sensitive_index])
                original_discrimination_knn = odds_difference
                sheet['F' + str(cell_number + 34)] = str(original_discrimination_knn)
                sheet['F' + str(cell_number + 35)] = str(acc_test_knn)

            save_res.append(one_res_s)

            workbook.save(os.path.join(c_root, 'analysis_CC_LR_OS_'+str(kfold_i)+'.xlsx'))
