
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import  numpy as np
from discrimination.calculate_discrimination import calculate_original_discrimination
import copy
# sylvaFile = r'E:\code\data\data\lung.mat'
# lung = scio.loadmat(sylvaFile)['lung']

# file_path = r'E:\code\HoeffdingTree\data\uci\adult\adult_bak.csv'
# file_path_test = r'E:\code\HoeffdingTree\data\uci\adult\adult_test.csv'
# file_path = '/code/HoeffdingTree/data/uci/adult/adult_bak.csv'
# file_path_test = '/code/HoeffdingTree/data/uci/adult/adult_test.csv'
# data_array, data_array_test, columns_name_index = load_data(file_path, file_path_test)
#
# SEED = 555
# np.random.seed(SEED)
# np.random.shuffle(data_array)
#
# data_array_test_copy = copy.deepcopy(data_array_test)
# class_index=data_array.shape[-1] -1
#
# testdata=data_array_test
# traindata=data_array



def set_sensitive_att_as_label(data_set, label_index, sensitive_index):
    data_set_remove_label = data_set[:, :label_index]
    sensitive_feature = data_set_remove_label[:, sensitive_index].reshape((-1,1))
    im_sensitive_feature = np.concatenate((data_set_remove_label[:, :sensitive_index] ,data_set_remove_label[:, sensitive_index + 1:]),axis=1)
    sensitive_label_dataset = np.concatenate((im_sensitive_feature, sensitive_feature),axis=1 )
    remove_sensitive_dataset = np.concatenate((im_sensitive_feature, data_set[:, label_index].reshape((-1,1))),axis=1 )
    two_label = np.concatenate((data_set[:, label_index].reshape((-1,1)), sensitive_feature),axis=1)
    return sensitive_label_dataset, remove_sensitive_dataset, im_sensitive_feature, two_label



def XGB_V2(im_sensitive_feature, two_label, im_sensitive_feature_test, two_label_test):
    Y = two_label
    X = im_sensitive_feature

    n_samples, n_features = X.shape  # 10,100
    n_outputs = Y.shape[1]
    n_classes = 2

    model = XGBClassifier(learning_rate=0.01,
                          n_estimators=10,
                          max_depth=4,
                          min_child_weight=1,
                          gamma=0.,
                          subsample=1,
                          colsample_btree=1,
                          scale_pos_weight=1,
                          random_state=27,
                          slient=0
                          )


    clf_model = model.fit(X, Y)

    acc_test = 0
    y_pred_test = clf_model.predict(im_sensitive_feature_test).reshape((-1,1))
    for i in range(two_label_test.shape[-1]):
        print("label i:", i)
        acc_0 = accuracy_score(two_label_test[:, i], y_pred_test[:, i])
        print("acc_O_test", acc_0)
        if i == 0:
            acc_test = acc_0

    return y_pred_test, acc_test




def XGB(im_sensitive_feature, two_label, im_sensitive_feature_test, two_label_test):
    Y = two_label
    X = im_sensitive_feature

    n_samples, n_features = X.shape  # 10,100
    n_outputs = Y.shape[1]
    n_classes = 2
    # forest = RandomForestClassifier(n_estimators=100, random_state=1)
    # multi_target_forest = MultiOutputClassifier(forest)

    model = XGBClassifier(learning_rate=0.01,
                          n_estimators=10,
                          max_depth=4,
                          min_child_weight=1,
                          gamma=0.,
                          subsample=1,
                          colsample_btree=1,
                          scale_pos_weight=1,
                          random_state=27,
                          slient=0
                          )


    clf_model = model.fit(X, Y)


    y_pred_test = clf_model.predict(im_sensitive_feature_test).reshape((-1,1))
    for i in range(two_label_test.shape[-1]):
        print("label i:", i)
        acc_0 = accuracy_score(two_label_test[:, i], y_pred_test[:, i])
        # acc_1 = accuracy_score(two_label_test[:, i], y_pred_test[:, i])
        print("acc_O_test", acc_0)

    return y_pred_test


def GNB(im_sensitive_feature, two_label, im_sensitive_feature_test, two_label_test):
    Y = two_label
    X = im_sensitive_feature

    n_samples, n_features = X.shape  # 10,100
    n_outputs = Y.shape[1]
    n_classes = 2


    from sklearn.naive_bayes import GaussianNB


    clf = GaussianNB()
    clf_model = clf.fit(X, Y)


    y_pred_test = clf_model.predict(im_sensitive_feature_test).reshape((-1,1))
    for i in range(two_label_test.shape[-1]):
        print("label i:", i)
        acc_0 = accuracy_score(two_label_test[:, i], y_pred_test[:, i])
        # acc_1 = accuracy_score(two_label_test[:, i], y_pred_test[:, i])
        print("acc_O_test", acc_0)

    return y_pred_test

def LR(im_sensitive_feature, two_label, im_sensitive_feature_test, two_label_test):
    Y = two_label
    X = im_sensitive_feature

    n_samples, n_features = X.shape  # 10,100
    n_outputs = Y.shape[1]
    n_classes = 2



    LR = LogisticRegression()
    LR_model = LR.fit(X, Y.astype(int))
    y_pred = LR_model.predict(X)


    y_pred_test = LR_model.predict(im_sensitive_feature_test).reshape((-1,1))
    for i in range(two_label_test.shape[-1]):
        print("label i:", i)
        acc_0 = accuracy_score(two_label_test[:, i].astype(int), y_pred_test[:, i])
        # acc_1 = accuracy_score(two_label_test[:, i], y_pred_test[:, i])
        print("acc_O_test", acc_0)

    return y_pred_test


from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict, StratifiedKFold
def LR_V2(im_sensitive_feature, two_label, im_sensitive_feature_test, two_label_test):
    Y = two_label
    X = im_sensitive_feature

    n_samples, n_features = X.shape  # 10,100
    n_outputs = Y.shape[1]
    n_classes = 2


    LR = LogisticRegression()
    LR_model = LR.fit(X, Y)
    y_pred = LR_model.predict(X)


    y_pred_test = LR_model.predict(im_sensitive_feature_test).reshape((-1, 1))
    for i in range(two_label_test.shape[-1]):
        print("label i:", i)
        acc_0 = accuracy_score(two_label_test[:, i], y_pred_test[:, i])
        # acc_1 = accuracy_score(two_label_test[:, i], y_pred_test[:, i])
        print("acc_O_test", acc_0)


    print("----------LogisticRegression------------")
    penaltys = ['l1', 'l2']
    Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    tuned_parameters = dict(penalty=penaltys, C=Cs)
    lr_penalty = LogisticRegression()
    X_train = X
    y_train = Y
    X_test = im_sensitive_feature_test
    grid = GridSearchCV(lr_penalty, tuned_parameters, cv=10, scoring=scoring)
    grid.fit(X_train, y_train)
    # grid.cv_results_
    model = grid.best_estimator_
    best_params = grid.best_params_


    y_pred_test = model.predict(X_test)
    for i in range(two_label_test.shape[-1]):
        print("label i:", i)
        acc_0 = accuracy_score(two_label_test[:, i], y_pred_test[:, i])

        print("acc_O_test", acc_0)

    return y_pred_test


def MLP(im_sensitive_feature, two_label, im_sensitive_feature_test, two_label_test, discrimination=False):
    Y = two_label
    X = im_sensitive_feature

    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(64, 32, 16), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=42, shuffle=True,
       solver='adam', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)
    from sklearn.multiclass import OneVsRestClassifier

    model = MultiOutputClassifier(model)  # 构建多输出多分类器
    multi_target_forest_model = model.fit(X, Y)

    y_pred_test = multi_target_forest_model.predict(im_sensitive_feature_test)
    if discrimination:
        original_discrimination = calculate_original_discrimination(class_label=y_pred_test[:, 0],
                                                                    discrimination_column=two_label_test[:, -1])
        print("after two label discrimination:", original_discrimination)
    for i in range(two_label_test.shape[-1]):
        print("label i:", i)
        acc_0 = accuracy_score(two_label_test[:, i], y_pred_test[:, i])
        print("acc_O_test", acc_0)

    return y_pred_test

