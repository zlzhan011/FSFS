from statistical_comparsion.knnclassify.knnclassify import knnclassify
from discrimination.calculate_discrimination import calculate_original_discrimination
from fairlearn.metrics import equalized_odds_difference

def knnclassify_calculate(selectedFeatures, data_array_train, data_array_test, n_neighbors=5):
    class_index = data_array_train.shape[-1] - 1
    y_predict, probability, acc_score, auc = knnclassify(data_array_train[:, selectedFeatures],
                                                         data_array_train[:, class_index],
                                                         data_array_test[:, selectedFeatures],
                                                         data_array_test[:, class_index], n_neighbors,
                                                         weights='uniform')

    original_discrimination = calculate_original_discrimination(class_label=y_predict,
                                                                discrimination_column=data_array_test[:, 9])
    print("original_discrimination:", original_discrimination)
    print("acc_score:", acc_score)

    odds_difference = equalized_odds_difference(data_array_test[:, class_index].reshape((-1, 1)),
                                                y_predict,
                                                sensitive_features=data_array_test[:, 9])
    print("odds_difference:", odds_difference)


    return y_predict, acc_score, original_discrimination


def knnclassify_calculate_v2(selectedFeatures, data_array_train, data_array_test, n_neighbors=5):
    class_index = data_array_train.shape[-1] - 1
    y_predict, probability, acc_score, auc = knnclassify(data_array_train[:, selectedFeatures],
                                                         data_array_train[:, class_index],
                                                         data_array_test[:, selectedFeatures],
                                                         data_array_test[:, class_index], n_neighbors,
                                                         weights='uniform')


    print("acc_score:", acc_score)

    return y_predict, acc_score
