
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.models import load_model
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score, f1_score
import  numpy as np
from keras import backend as K






def get_label_value(predict):

    if len(predict[0]) == 2:
        predict_res = []
        for i in range(len(predict)):
            label_1 = np.argmax(predict[i])
            predict_res.append([label_1])
    elif  len(predict[0]) == 1:
        predict_res = []
        for i in range(len(predict)):
            if predict[i] >= 0.5:
                predict_res.append(1)
            else:
                predict_res.append(0)
    else:
        predict_res = []
        for i in range(len(predict)):
            label_1 = np.argmax(predict[i][:-2])
            label_2 = np.argmax(predict[i][-2:])
            predict_res.append([label_1, label_2])
    return np.array(predict_res)


def evaluate_MLP_Keras(model_path, X_test, y_test):
    model = load_model(model_path)
    y_test_predict = model.predict(X_test)
    print("y_test_predict:", y_test_predict[0])
    print("y_test:", y_test[0])



    y_test_predict = get_label_value(y_test_predict)
    y_test = get_label_value(y_test)

    if len(y_test_predict.shape) == 2:
        for i in range(y_test_predict.shape[-1]):
            print("label i:", i)
            acc_0 = accuracy_score(y_test_predict[:, i], y_test[:, i])
            print("acc_0:", acc_0)
    else:
        acc_0 = accuracy_score(y_test_predict, y_test)
        print("acc_0:", acc_0)

    return y_test_predict


def MLP_Keras_V2(X_train, y_train, X_test,y_test, model_path='mlp_keras_model.pth', convert_two_label_flag = True):
    seed_value = 555
    tf.random.set_seed(seed_value)
    np.random.seed(seed_value)
    if convert_two_label_flag:
        y_train = convert_two_label(y_train)
        y_test = convert_two_label(y_test)
    # X_train, y_train, X_test, y_test = get_MINST()
    # 创建模型
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=X_train.shape[-1]))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y_train.shape[-1], activation='sigmoid'))
    # model.add(Dense(12, activation='softmax'))
    # inputs = Input(shape=(10,))
    # hidden = Dense(units=10,activation='relu')(inputs)
    # output = Dense(units=5,activation='sigmoid')(hidden)
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 编译模型
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy","Precision","Recall"])

    # 训练模型
    model.fit(X_train.astype(np.float), y_train, epochs=50, batch_size=32,verbose=0)
    model.save(model_path)
    y_test_predict = model.predict(X_test.astype(np.float))
    y_test_predict = get_label_value(y_test_predict)
    y_test = get_label_value(y_test)

    y_train_predict = model.predict(X_train.astype(np.float))
    y_train_predict = get_label_value(y_train_predict)
    y_train = get_label_value(y_train)




    if len(y_test_predict.shape) == 2:
        for i in range(y_test_predict.shape[-1]):
            print("label i:", i)
            acc_0_test = accuracy_score(y_test_predict[:, i], y_test[:, i])
            f1_0 = f1_score(y_test_predict[:, i], y_test[:, i])
            p_0 = precision_score(y_test_predict[:, i], y_test[:, i])
            r_0 = recall_score(y_test_predict[:, i], y_test[:, i])
            print("test acc_0:", acc_0_test)
            print("test f1_0:", f1_0)
            print("test p_0:", p_0)
            print("test r_0:", r_0)
            acc_0_train = accuracy_score(y_train_predict[:, i], y_train[:, i])
            f1_0 = f1_score(y_train_predict[:, i], y_train[:, i])
            p_0 = precision_score(y_train_predict[:, i], y_train[:, i])
            r_0 = recall_score(y_train_predict[:, i], y_train[:, i])
            print("train acc_0:", acc_0_train)
            print("train f1_0:", f1_0)
            print("train p_0:", p_0)
            print("train r_0:", r_0)
    else:
        acc_0_test = accuracy_score(y_test_predict, y_test)
        f1_0 = f1_score(y_test_predict, y_test)
        p_0 = precision_score(y_test_predict, y_test)
        r_0 = recall_score(y_test_predict, y_test)
        print("test acc_0:", acc_0_test)
        print("test f1_0:", f1_0)
        print("test p_0:", p_0)
        print("test r_0:", r_0)
        acc_0_train = accuracy_score(y_train_predict, y_train)
        f1_0 = f1_score(y_train_predict, y_train)
        p_0 = precision_score(y_train_predict, y_train)
        r_0 = recall_score(y_train_predict, y_train)
        print("train acc_0:", acc_0_train)
        print("train f1_0:", f1_0)
        print("train p_0:", p_0)
        print("train r_0:", r_0)
    del model
    return y_test_predict, acc_0_test

def MLP_Keras(X_train, y_train, X_test,y_test, model_path='mlp_keras_model.pth', convert_two_label_flag = True):
    seed_value = 555
    tf.random.set_seed(seed_value)
    np.random.seed(seed_value)
    if convert_two_label_flag:
        y_train = convert_two_label(y_train)
        y_test = convert_two_label(y_test)
    # X_train, y_train, X_test, y_test = get_MINST()
    # 创建模型
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=X_train.shape[-1]))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y_train.shape[-1], activation='sigmoid'))
    # model.add(Dense(12, activation='softmax'))
    # inputs = Input(shape=(10,))
    # hidden = Dense(units=10,activation='relu')(inputs)
    # output = Dense(units=5,activation='sigmoid')(hidden)
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 编译模型
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy","Precision","Recall"])

    # 训练模型
    model.fit(X_train.astype(np.float), y_train, epochs=50, batch_size=32,verbose=0)
    model.save(model_path)
    y_test_predict = model.predict(X_test.astype(np.float))
    y_test_predict = get_label_value(y_test_predict)
    y_test = get_label_value(y_test)

    y_train_predict = model.predict(X_train.astype(np.float))
    y_train_predict = get_label_value(y_train_predict)
    y_train = get_label_value(y_train)




    if len(y_test_predict.shape) == 2:
        for i in range(y_test_predict.shape[-1]):
            print("label i:", i)
            acc_0 = accuracy_score(y_test_predict[:, i], y_test[:, i])
            f1_0 = f1_score(y_test_predict[:, i], y_test[:, i])
            p_0 = precision_score(y_test_predict[:, i], y_test[:, i])
            r_0 = recall_score(y_test_predict[:, i], y_test[:, i])
            print("test acc_0:", acc_0)
            print("test f1_0:", f1_0)
            print("test p_0:", p_0)
            print("test r_0:", r_0)
            acc_0 = accuracy_score(y_train_predict[:, i], y_train[:, i])
            f1_0 = f1_score(y_train_predict[:, i], y_train[:, i])
            p_0 = precision_score(y_train_predict[:, i], y_train[:, i])
            r_0 = recall_score(y_train_predict[:, i], y_train[:, i])
            print("train acc_0:", acc_0)
            print("train f1_0:", f1_0)
            print("train p_0:", p_0)
            print("train r_0:", r_0)
    else:
        acc_0 = accuracy_score(y_test_predict, y_test)
        f1_0 = f1_score(y_test_predict, y_test)
        p_0 = precision_score(y_test_predict, y_test)
        r_0 = recall_score(y_test_predict, y_test)
        print("test acc_0:", acc_0)
        print("test f1_0:", f1_0)
        print("test p_0:", p_0)
        print("test r_0:", r_0)
        acc_0 = accuracy_score(y_train_predict, y_train)
        f1_0 = f1_score(y_train_predict, y_train)
        p_0 = precision_score(y_train_predict, y_train)
        r_0 = recall_score(y_train_predict, y_train)
        print("train acc_0:", acc_0)
        print("train f1_0:", f1_0)
        print("train p_0:", p_0)
        print("train r_0:", r_0)
    del model
    return y_test_predict




def convert_two_label(two_label):
    # two_label_0 = to_categorical(two_label[:, 0])
    # two_label_1 = to_categorical(two_label[:, 1])
    if len(two_label[0]) == 2:
        return np.c_[to_categorical(two_label[:, 0]), to_categorical(two_label[:, 1])]
    else:
        return to_categorical(two_label[:, 0])
#
# if __name__ == '__main__':
#     # X_train, y_train, X_test, y_test = get_MINST()
#
#     file_path = r'E:\code\HoeffdingTree\data\uci\adult\adult_bak.csv'
#     file_path_test = r'E:\code\HoeffdingTree\data\uci\adult\adult_test.csv'
#     data_array, data_array_test, columns_name_index = load_data(file_path, file_path_test)
#
#     SEED = 555
#     np.random.seed(SEED)
#     np.random.shuffle(data_array)
#
#     data_array_test_copy = copy.deepcopy(data_array_test)
#     class_index = data_array.shape[-1] - 1
#
#     testdata = data_array_test
#     traindata = data_array
#     sensitive_label_dataset, remove_sensitive_dataset, im_sensitive_feature, two_label= set_sensitive_att_as_label(traindata, class_index, 9)
#     sensitive_label_dataset_test, remove_sensitive_dataset_test, im_sensitive_feature_test, two_label_test= set_sensitive_att_as_label(testdata, class_index, 9)
#
#     two_label_copy = copy.deepcopy(two_label)
#     two_label_test_copy = copy.deepcopy(two_label_test)
#
#     im_sensitive_feature_test_copy = copy.deepcopy(im_sensitive_feature_test)
#     im_sensitive_feature_copy = copy.deepcopy(im_sensitive_feature)
#     # im_sensitive_feature, two_label
#     im_sensitive_feature, im_sensitive_feature_test = convert_one_hot(im_sensitive_feature_test, im_sensitive_feature, im_sensitive_feature_test_copy,
#                     im_sensitive_feature_copy)
#     y_test_predict = MLP_Keras(X_train = im_sensitive_feature, y_train =two_label, X_test = im_sensitive_feature_test, y_test=two_label_test)
#     original_discrimination = calculate_original_discrimination(class_label=y_test_predict[:, 0],
#                                                                     discrimination_column=two_label_test_copy[:, -1])
#     print("original_discrimination:", original_discrimination)
#
#     """
#
#
#
#     label i: 0
#     acc_0: 0.7882808181315644
#     label i: 1
#     acc_0: 0.6670351943983784
#     favored_granted: 433 --favored_rejected: 10427 --deprived_granted: 110 --deprived_rejected: 5311
#     original_discrimination: 0.019579627415789426
#
#
#
#
#
#     MNIST dataset
#     label
#     i: 0
#     acc_0: 0.9722
#     label
#     i: 1
#     acc_0: 0.9824
#
#
#
#
#
#     """
