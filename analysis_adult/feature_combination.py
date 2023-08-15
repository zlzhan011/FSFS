import numpy as np
from sklearn.preprocessing import PolynomialFeatures
# X = np.arange(6).reshape(3, 2)
# print(X.shape)
# print(X)
# poly = PolynomialFeatures(3)
# X_1 = poly.fit_transform(X)
#
# print(X_1.shape)
# print(X_1)
#
#
# poly = PolynomialFeatures(degree=3, interaction_only=True)
# X_2 = poly.fit_transform(X)
# print(X_2.shape)
# print(X_2)


def feature_combine(data_array):
    data_array_X = data_array[:, :-1]
    poly = PolynomialFeatures(3)
    data_array_X = poly.fit_transform(data_array_X)

    data_array = np.concatenate((data_array_X, data_array[:, -1].reshape((-1, 1))), axis=1)
    return data_array



def set_sensitive_att_as_label(data_set, label_index, sensitive_index):
    data_set_remove_label = data_set[:, :label_index]
    sensitive_feature = data_set_remove_label[:, sensitive_index].reshape((-1,1))
    im_sensitive_feature = np.concatenate((data_set_remove_label[:, :sensitive_index] ,data_set_remove_label[:, sensitive_index + 1:]),axis=1)
    sensitive_label_dataset = np.concatenate((im_sensitive_feature, sensitive_feature),axis=1 )
    remove_sensitive_dataset = np.concatenate((im_sensitive_feature, data_set[:, label_index].reshape((-1,1))),axis=1 )
    return sensitive_label_dataset, remove_sensitive_dataset