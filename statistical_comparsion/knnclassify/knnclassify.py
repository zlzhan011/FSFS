# -*-coding:utf-8 -*-
from sklearn import datasets
import  sklearn
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def knnclassify(x_train, y_train, x_test, y_test, n_neighbors, weights='uniform'):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    # clf = KNeighborsClassifier(n_neighbors, weights=weights)
    # 定义一个knn分类器对象
    knn.fit(x_train, y_train)
    # 调用该对象的训练方法，主要接收两个参数：训练数据集及其样本标签

    y_predict = knn.predict(x_test)
    # 调用该对象的测试方法，主要接收一个参数：测试数据集
    probility = knn.predict_proba(x_test)
    # # 计算各测试样本基于概率的预测
    # neighborpoint = knn.kneighbors(x_test[-1], 5, False)
    # 计算与最后一个测试样本距离在最近的5个点，返回的是这些样本的序号组成的数组
    acc_score = knn.score(x_test, y_test, sample_weight=None)
    # 调用该对象的打分方法，计算出准确率
    # auc = sklearn.metrics.roc_auc_score(y_test, probility[:,-1])
    auc = ''
    return y_predict, probility, acc_score, auc
