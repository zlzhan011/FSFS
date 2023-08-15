import os.path

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import math
import  json

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from plotly.subplots import make_subplots
from sklearn.model_selection import  StratifiedKFold

from lightgbm import LGBMClassifier
fig = make_subplots(rows=2, cols=2, specs=[[{}, {}], [{"colspan": 2}, None]], subplot_titles=("Males", "Females", "All Genders"))
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")
import random
seed = 42
random.seed(seed)
np.random.seed(seed)




from sklearn.metrics import precision_recall_curve, roc_curve, accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score


def metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)

    return accuracy, precision, recall, f1, roc_auc


def print_metrics(y_true, y_pred):
    accuracy, precision, recall, f1_score, roc_auc = metrics(y_true, y_pred)
    print("Accuracy: %.3f\nPrecision: %.3f\nRecall: %.3f\nF1 Score: %.3f\nROC AUC: %.3f" % (
    accuracy, precision, recall, f1_score, roc_auc))
    return accuracy


def plot_curves(y_true, probas):
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    precision, recall, thresholds = precision_recall_curve(y_true, probas)
    plt.plot(recall, precision, color="b")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.subplot(122)
    fpr, tpr, tresholds = roc_curve(y_true, probas)
    plt.plot(fpr, tpr, color="g")
    plt.plot([0, 1], [0, 1], color="black", linestyle="--")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    fig.show()


kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)




def write_res(res, file_path):
    with  open(file_path, 'w', encoding='utf8') as f_write:
        json.dump(res,f_write)




def modeling_no_knn(X_train, y_train, X_test, y_test, write_flag="", c_root=""):
    import warnings
    warnings.filterwarnings("ignore")
    from sklearn.exceptions import DataConversionWarning
    warnings.filterwarnings("ignore", category=DataConversionWarning)
    models = {
        "LR": LogisticRegression(random_state=seed, max_iter=10000),
        # "CART": DecisionTreeClassifier(random_state=seed),
        # "NB": GaussianNB(),
        # "KNN": KNeighborsClassifier(),
        # "RF": RandomForestClassifier(random_state=seed),
        # "SVM": SVC(random_state=seed),
        # "XGB": XGBClassifier(),
        # "LGBM": LGBMClassifier(),
        # "MLP": MLPClassifier(random_state=seed)
    }
    res = {}
    # scoring = "recall"
    scoring = "accuracy"
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    scores = []

    # for name, model in models.items():
    #     _scores = cross_val_score(model, X_train, y_train, scoring=scoring, cv=kfold, n_jobs=-1, verbose=1)
    #     print("name: ", name)
    #     print(_scores)
    #     msg = scoring + " %s has an average score of %.3f ± %.3f" % (name, np.mean(_scores), np.std(_scores))
    #     print(msg)
    #     scores.append(_scores)


    # scores_df = pd.DataFrame(data=np.array(scores), index=list(models.keys())).reset_index().rename(
    #     columns=dict(index="model"))
    # scores_df = pd.melt(scores_df, id_vars=["model"], value_vars=np.arange(0, 10)).rename(
    #     columns=dict(variable="fold", value="score"))
    #
    # plt.figure(figsize=(4, 4))
    # sns.boxplot(data=scores_df, x="model", y="score")
    # plt.title("Cross-validated Model Accuracy Scores")
    # plt.ylim((0, 1))
    # # plt.show()
    # plt.savefig(os.path.join(c_root, r'cv_'+write_flag+'_.jpg'))
    import time
    time_start = time.time()
    print("----------LogisticRegression------------")
    penaltys = ['l1', 'l2']
    Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    tuned_parameters = dict(penalty=penaltys, C=Cs)
    # neg_log_loss
    lr_penalty = LogisticRegression()
    grid = GridSearchCV(lr_penalty, tuned_parameters, cv=10, scoring=scoring)
    grid.fit(X_train, y_train)
    # grid.cv_results_
    model = grid.best_estimator_
    best_params = grid.best_params_
    print("best_params:", best_params)

    y_pre = model.predict(X_test)
    accuracy = print_metrics(y_test, y_pre)
    res['MLP'] = {"best_params":best_params,
                 "y_pre":y_pre.astype(int).tolist(),
                 "accuracy":accuracy}
    time_end = time.time()
    print("LR time cost:", time_end - time_start)





    # print("----------GaussianNB------------")
    # model = GaussianNB()
    # model.fit(X_train, y_train)
    # probas = model.predict_proba(X_test)
    # y_pre = model.predict(X_test)
    # accuracy = print_metrics(y_test, y_pre)
    # # plot_curves(y_test, probas[:, 1])
    #
    # res['NB'] = {"y_pre": y_pre.astype(int).tolist(),
    #              "accuracy": accuracy}

    # In[ ]:

    # print("----------DecisionTreeClassifier------------")
    #
    # scoring= 'accuracy'
    # model = DecisionTreeClassifier(random_state=seed)
    # gscv = GridSearchCV(
    #     model,
    #     param_grid={
    #         "criterion": ["gini", "entropy", "log_loss"],
    #         "max_depth": [i for i in range(2,11)],
    #         "min_samples_split": [i for i in range(2,101, 10)],
    #         "max_features": ["sqrt", "log2", None]
    #     },
    #     scoring=scoring,
    #     cv=kfold,
    #     n_jobs=-1
    # )
    # gscv.fit(X_train, y_train)
    # best_model = gscv.best_estimator_
    # best_params = gscv.best_params_
    # print("best_params:", best_params)
    # probas = best_model.predict_proba(X_test)
    # y_pre = best_model.predict(X_test)
    # accuracy = print_metrics(y_test, y_pre)
    # # plot_curves(y_test, probas[:, 1])
    #
    # res['DT'] = {"best_params": best_params,
    #              "y_pre": y_pre.astype(int).tolist(),
    #              "accuracy": accuracy}

    # In[ ]:

    # print("----------XGBClassifier------------")
    # model = XGBClassifier()
    # gscv = GridSearchCV(
    #     model,
    #     param_grid={
    #         "max_depth": [i for i in range(2,11)]
    #     },
    #     scoring=scoring,
    #     cv=kfold,
    #     n_jobs=-1
    # )
    # gscv.fit(X_train, y_train)
    # best_model = gscv.best_estimator_
    # best_params = gscv.best_params_
    # print("best_params:", best_params)
    # probas = best_model.predict_proba(X_test)
    # y_pre = best_model.predict(X_test)
    # accuracy = print_metrics(y_test, y_pre)
    # # plot_curves(y_test, probas[:, 1])
    # res['XGB'] = {"best_params": best_params,
    #              "y_pre": y_pre.astype(int).tolist(),
    #              "accuracy": accuracy}

    # ### <a id="lgbm"></a>LightGBM

    # In[ ]:

    print("----------LGBMClassifier------------")
    model = LGBMClassifier()
    gscv = GridSearchCV(
        model,
        param_grid={
            "max_depth": [i for i in range(2,11)]
        },
        scoring=scoring,
        cv=kfold,
        n_jobs=-1
    )
    gscv.fit(X_train, y_train)
    best_model = gscv.best_estimator_
    best_params = gscv.best_params_
    print("best_params:", best_params)
    probas = best_model.predict_proba(X_test)
    y_pre = best_model.predict(X_test)
    accuracy = print_metrics(y_test, y_pre)
    # plot_curves(y_test, probas[:, 1])
    res['LGBM'] = {"best_params": best_params,
                  "y_pre": y_pre.astype(int).tolist(),
                  "accuracy": accuracy}


    # mlp_cv=True
    # if mlp_cv:
    #     print("----------MLP------------")
    #     import  time
    #     time_start = time.time()
    #     param_grid = {
    #         # 'hidden_layer_sizes': [(50,), (100,), (50, 50),(64, 32), (128, 32)],
    #         'hidden_layer_sizes': [(1000,50)],
    #         # 'activation': ['relu', 'tanh'],
    #         # 'activation': ['relu'],
    #
    #         'solver': ['sgd', 'adam'],
    #         # 'solver': ['sgd'],
    #         # 'alpha': [0.001, 0.01],
    #         # 'dropout': [0.2, 0.35, 0.5]
    #     }
    #
    #
    #     model = MLPClassifier()
    #
    #     from sklearn.preprocessing import StandardScaler
    #
    #
    #     grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    #     print(" MLP MLP MLP MLP--------------------------------")
    #     from imblearn.over_sampling import SMOTE
    #     sm = SMOTE(random_state=42)
    #     X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
    #     X_train = X_resampled
    #     y_train = y_resampled
    #     # scaler = StandardScaler()
    #     # X_train = scaler.fit_transform(X_train)
    #     # X_test = scaler.fit_transform(X_test)
    #
    #
    #
    #
    #
    #     grid_search.fit(X_train, y_train)
    #
    #
    #
    #     best_model = grid_search.best_estimator_
    #     best_params = grid_search.best_params_
    #     print("best_params:", best_params)
    #     accuracy = best_model.score(X_test, y_test)
    #     print("accuracy:", accuracy)
    #     y_pre = best_model.predict(X_test)
    #     accuracy = print_metrics(y_test, y_pre)
    #     res['MLP'] = {"best_params": best_params,
    #                   "y_pre": y_pre.astype(int).tolist(),
    #                   "accuracy": accuracy}
    #
    #     time_end = time.time()
    #     print("MLP cost time:", time_end - time_start)
    # else:
    #     print("----------MLP------------")
    #     import time
    #     time_start = time.time()
    #     param_grid = {
    #         # 'hidden_layer_sizes': [(50,), (100,), (50, 50),(64, 32), (128, 32)],
    #         'hidden_layer_sizes': [(1000,50)],
    #         # 'activation': ['relu', 'tanh'],
    #         'activation': ['relu'],
    #
    #         # 'solver': ['sgd', 'adam'],
    #         'solver': ['sgd'],
    #         # 'alpha': [0.001, 0.01],
    #         # 'dropout': [0.2, 0.35, 0.5]
    #     }
    #
    #     model = MLPClassifier(hidden_layer_sizes=(1000,50),
    #     activation="relu",
    #     solver="adam",
    #     alpha=0.0001,
    #     batch_size="auto",
    #     learning_rate="constant",
    #     learning_rate_init=0.001,
    #     power_t=0.5,
    #     max_iter=200,
    #     shuffle=True,
    #     random_state=None,
    #     tol=1e-4,
    #     verbose=False,
    #     warm_start=False,
    #     momentum=0.9,
    #     nesterovs_momentum=True,
    #     early_stopping=False,
    #     validation_fraction=0.1,
    #     beta_1=0.9,
    #     beta_2=0.999,
    #     epsilon=1e-8,
    #     n_iter_no_change=10,
    #     max_fun=15000,)
    #
    #     from sklearn.preprocessing import StandardScaler
    #
    #     # grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    #     grid_search = model
    #     print(" MLP MLP MLP MLP---no cv-----------------------------")
    #     # from imblearn.over_sampling import SMOTE
    #     # sm = SMOTE(random_state=42)
    #     # X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
    #     # X_train = X_resampled
    #     # y_train = y_resampled
    #     # scaler = StandardScaler()
    #     # X_train = scaler.fit_transform(X_train)
    #     # X_test = scaler.fit_transform(X_test)
    #
    #     grid_search.fit(X_train, y_train)
    #
    #     # best_model = grid_search.best_estimator_
    #     # best_params = grid_search.best_params_
    #     # print("best_params:", best_params)
    #     best_model = grid_search
    #     accuracy = best_model.score(X_test, y_test)
    #     print("accuracy:", accuracy)
    #     y_pre = best_model.predict(X_test)
    #     accuracy = print_metrics(y_test, y_pre)
    #     res['MLP'] = {"best_params": best_params,
    #                   "y_pre": y_pre.astype(int).tolist(),
    #                   "accuracy": accuracy}
    #
    #     time_end = time.time()
    #     print("MLP cost time:", time_end - time_start)




    # print("----------KNN------------")
    # # 定义参数网格
    # param_grid = {
    #     'n_neighbors': [1, 3, 5, 7],
    #     'weights': ['uniform', 'distance'],
    # }

    # model = KNeighborsClassifier()
    #
    # grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    # grid_search.fit(X_train, y_train)
    #
    # best_model = grid_search.best_estimator_
    # best_params = grid_search.best_params_
    # print("best_params:", best_params)
    # accuracy = best_model.score(X_test, y_test)
    # y_pre = best_model.predict(X_test)
    # accuracy = print_metrics(y_test, y_pre)
    # res['KNN'] = {"best_params": best_params,
    #               "y_pre": y_pre.astype(int).tolist(),
    #               "accuracy": accuracy}
    res['shape'] = X_train.shape
    print("X_train_shape:", X_train.shape)
    write_res(res, os.path.join(c_root, r'res_'+write_flag+'_.json'))
    return res




def modeling(X_train, y_train, X_test, y_test, write_flag="", c_root=""):
    models = {
        "LR": LogisticRegression(random_state=seed, max_iter=10000),
        # "CART": DecisionTreeClassifier(random_state=seed),
        # "NB": GaussianNB(),
        # "KNN": KNeighborsClassifier(),
        # "RF": RandomForestClassifier(random_state=seed),
        # "SVM": SVC(random_state=seed),
        # "XGB": XGBClassifier(),
        "LGBM": LGBMClassifier(),
        # "MLP": MLPClassifier(random_state=seed)
    }
    res = {}
    # scoring = "recall"
    scoring = "accuracy"
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    scores = []

    for name, model in models.items():
        _scores = cross_val_score(model, X_train, y_train, scoring=scoring, cv=kfold, n_jobs=-1, verbose=1)
        print("name: ", name)
        print(_scores)
        msg = scoring + " %s has an average score of %.3f ± %.3f" % (name, np.mean(_scores), np.std(_scores))
        print(msg)
        scores.append(_scores)


    scores_df = pd.DataFrame(data=np.array(scores), index=list(models.keys())).reset_index().rename(
        columns=dict(index="model"))
    scores_df = pd.melt(scores_df, id_vars=["model"], value_vars=np.arange(0, 10)).rename(
        columns=dict(variable="fold", value="score"))

    plt.figure(figsize=(4, 4))
    sns.boxplot(data=scores_df, x="model", y="score")
    plt.title("Cross-validated Model Accuracy Scores")
    plt.ylim((0, 1))
    # plt.show()
    plt.savefig(os.path.join(c_root, r'cv_'+write_flag+'_.jpg'))
    import  time
    time_lr = time.time()
    print("----------LogisticRegression------------")
    penaltys = ['l1', 'l2']
    Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    tuned_parameters = dict(penalty=penaltys, C=Cs)
    # neg_log_loss
    lr_penalty = LogisticRegression()
    grid = GridSearchCV(lr_penalty, tuned_parameters, cv=10, scoring=scoring)
    grid.fit(X_train, y_train)
    # grid.cv_results_
    model = grid.best_estimator_
    best_params = grid.best_params_
    print("best_params:", best_params)

    y_pre = model.predict(X_test)
    accuracy = print_metrics(y_test, y_pre)
    res['MLP'] = {"best_params":best_params,
                 "y_pre":y_pre.astype(int).tolist(),
                 "accuracy":accuracy}

    time_lr_end = time.time()
    print("LR time cost:", time_lr_end - time_lr)



    # from sklearn.pipeline import Pipeline
    # from sklearn.preprocessing import MinMaxScaler
    # from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
    # numerical_transformer = Pipeline(steps=[('scaler', MinMaxScaler())])
    # categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder(drop='if_binary'))])
    #
    # features2_scale = []
    # features2_encode = []
    #
    # # target = 'is_recid'
    # # COMPAS = X_train
    # # for i in range(COMPAS.shape[-1]):
    # #     if COMPAS[:, i].dtype in (int, float) and i != target:
    # #         features2_scale.append(i)
    # #     elif i != target:
    # #         features2_encode.append(i)
    #
    # features2_encode = [i for i in range(X_train.shape[-1])]
    #
    # preprocessor = ColumnTransformer(
    #     transformers=[
    #         ('scale', numerical_transformer, features2_scale),
    #         ('encode', categorical_transformer, features2_encode)],
    #     remainder='passthrough')
    #
    # LR_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
    #                                    ('clf',
    #                                     LogisticRegression(penalty='l2', C=1, solver='lbfgs',
    #                                                        max_iter=100000, multi_class='auto'))])
    # print("----------LogisticRegression------------")
    # penaltys = ['l1', 'l2']
    # Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    # tuned_parameters = dict(penalty=penaltys, C=Cs)
    # # neg_log_loss
    # lr_penalty = LogisticRegression()
    # grid = GridSearchCV(lr_penalty, tuned_parameters, cv=10, scoring=scoring)
    # grid = LR_pipeline
    # grid.fit(X_train, y_train)
    # # grid.cv_results_
    # # model = grid.best_estimator_
    # # best_params = grid.best_params_
    # # print("best_params:", best_params)
    # model = LR_pipeline
    # y_pre = model.predict(X_test)
    # accuracy = print_metrics(y_test, y_pre)
    # res['MLP'] = {
    #               "y_pre": y_pre.astype(int).tolist(),
    #               "accuracy": accuracy}




    # print("----------GaussianNB------------")
    # model = GaussianNB()
    # model.fit(X_train, y_train)
    # probas = model.predict_proba(X_test)
    # y_pre = model.predict(X_test)
    # accuracy = print_metrics(y_test, y_pre)
    # # plot_curves(y_test, probas[:, 1])
    #
    # res['NB'] = {"y_pre": y_pre.astype(int).tolist(),
    #              "accuracy": accuracy}

    # In[ ]:

    # print("----------DecisionTreeClassifier------------")
    #
    # scoring= 'accuracy'
    # model = DecisionTreeClassifier(random_state=seed)
    # gscv = GridSearchCV(
    #     model,
    #     param_grid={
    #         "criterion": ["gini", "entropy", "log_loss"],
    #         "max_depth": [i for i in range(2,11)],
    #         "min_samples_split": [i for i in range(2,101, 10)],
    #         "max_features": ["sqrt", "log2", None]
    #     },
    #     scoring=scoring,
    #     cv=kfold,
    #     n_jobs=-1
    # )
    # gscv.fit(X_train, y_train)
    # best_model = gscv.best_estimator_
    # best_params = gscv.best_params_
    # print("best_params:", best_params)
    # probas = best_model.predict_proba(X_test)
    # y_pre = best_model.predict(X_test)
    # accuracy = print_metrics(y_test, y_pre)
    # # plot_curves(y_test, probas[:, 1])
    #
    # res['DT'] = {"best_params": best_params,
    #              "y_pre": y_pre.astype(int).tolist(),
    #              "accuracy": accuracy}

    # In[ ]:

    # print("----------XGBClassifier------------")
    # model = XGBClassifier()
    # gscv = GridSearchCV(
    #     model,
    #     param_grid={
    #         "max_depth": [i for i in range(2,11)]
    #     },
    #     scoring=scoring,
    #     cv=kfold,
    #     n_jobs=-1
    # )
    # gscv.fit(X_train, y_train)
    # best_model = gscv.best_estimator_
    # best_params = gscv.best_params_
    # print("best_params:", best_params)
    # probas = best_model.predict_proba(X_test)
    # y_pre = best_model.predict(X_test)
    # accuracy = print_metrics(y_test, y_pre)
    # # plot_curves(y_test, probas[:, 1])
    # res['XGB'] = {"best_params": best_params,
    #              "y_pre": y_pre.astype(int).tolist(),
    #              "accuracy": accuracy}

    # ### <a id="lgbm"></a>LightGBM

    # In[ ]:

    print("----------LGBMClassifier------------")
    model = LGBMClassifier()
    gscv = GridSearchCV(
        model,
        param_grid={
            "max_depth": [i for i in range(2,11)]
        },
        scoring=scoring,
        cv=kfold,
        n_jobs=-1
    )
    gscv.fit(X_train, y_train)
    best_model = gscv.best_estimator_
    best_params = gscv.best_params_
    print("best_params:", best_params)
    probas = best_model.predict_proba(X_test)
    y_pre = best_model.predict(X_test)
    accuracy = print_metrics(y_test, y_pre)
    # plot_curves(y_test, probas[:, 1])
    res['LGBM'] = {"best_params": best_params,
                  "y_pre": y_pre.astype(int).tolist(),
                  "accuracy": accuracy}

    import  time
    time_start = time.time()
    print("----------MLP------------")
    param_grid = {
        # 'hidden_layer_sizes': [(50,), (100,), (50, 50),(64, 32), (128, 32)],
        'hidden_layer_sizes': [(64, 32)],
        'activation': ['relu', 'tanh'],
        'solver': ['sgd', 'adam'],
    }


    # model = MLPClassifier()
    #
    #
    # grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10)
    # grid_search.fit(X_train, y_train)
    #
    #
    # best_model = grid_search.best_estimator_
    # best_params = grid_search.best_params_
    # print("best_params:", best_params)
    # accuracy = best_model.score(X_test, y_test)
    # print("accuracy:", accuracy)
    # y_pre = best_model.predict(X_test)
    # accuracy = print_metrics(y_test, y_pre)
    # res['MLP'] = {"best_params": best_params,
    #               "y_pre": y_pre.astype(int).tolist(),
    #               "accuracy": accuracy}
    #
    # time_end = time.time()
    # print("MLP cost time:", time_end - time_start)


    # print("----------KNN------------")
    # # 定义参数网格
    # param_grid = {
    #     'n_neighbors': [1, 3, 5, 7],
    #     'weights': ['uniform', 'distance'],
    # }

    # model = KNeighborsClassifier()
    #
    # grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    # grid_search.fit(X_train, y_train)
    #
    # best_model = grid_search.best_estimator_
    # best_params = grid_search.best_params_
    # print("best_params:", best_params)
    # accuracy = best_model.score(X_test, y_test)
    # y_pre = best_model.predict(X_test)
    # accuracy = print_metrics(y_test, y_pre)
    # res['KNN'] = {"best_params": best_params,
    #               "y_pre": y_pre.astype(int).tolist(),
    #               "accuracy": accuracy}
    res['shape'] = X_train.shape
    print("X_train_shape:", X_train.shape)
    write_res(res, os.path.join(c_root, r'res_'+write_flag+'_.json'))
    return res


