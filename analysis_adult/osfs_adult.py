
from learning_module.osfs_and_fast_osfs.osfs_z import osfs_z
from statistical_comparsion.knnclassify.knnclassify import knnclassify
import  numpy as np
from analysis_adult.read_adult import load_data
from discrimination.calculate_discrimination import calculate_original_discrimination
import copy


file_path = r'E:\code\HoeffdingTree\data\uci\adult\adult_bak.csv'
file_path_test = r'E:\code\HoeffdingTree\data\uci\adult\adult_test.csv'
data_array, data_array_test, columns_name_index = load_data(file_path, file_path_test)



SEED = 555
np.random.seed(SEED)
np.random.shuffle(data_array)

data_array_test_copy = copy.deepcopy(data_array_test)
# data_array = feature_combine(data_array)
# data_array_test = feature_combine(data_array_test)

data_array = data_array[:1000,:]

# selectedFeatures, time_g2 =osfs_d(lung,3312,0.01,'g2')





testdata=data_array_test
traindata=data_array
# testdata = traindata
class_index=data_array.shape[-1] -1
import time
time_start = time.time()
print("traindata_shape:", traindata.shape)
selectedFeatures, time22=osfs_z(traindata,class_index,0.01, max_k=100)
print("selectedFeatures:", selectedFeatures)
time_end = time.time()
print("time22:", time22)
print("cost time:", time_end - time_start)
print(data_array.shape)
print("selectedFeatures:", selectedFeatures)






# selectedFeatures = [i for i in range(class_index)]
y_predict, probility, acc_score, auc = knnclassify(traindata[:,selectedFeatures], traindata[:,class_index], testdata[:,selectedFeatures], testdata[:,class_index], 5, weights='uniform')
print("acc_score:", acc_score)
print("auc:", auc)
original_discrimination = calculate_original_discrimination(class_label=data_array_test[:,-1], discrimination_column=data_array_test_copy[:,9])
print("original_discrimination:", original_discrimination)
original_discrimination = calculate_original_discrimination(class_label=y_predict, discrimination_column=data_array_test_copy[:,9])
print("selectedFeatures discrimination:", original_discrimination)

selectedFeatures = [i for i in range(class_index)]
print("selectedFeatures  all :", selectedFeatures)


# selectedFeatures = [i for i in range(class_index)]
y_predict, probility, acc_score, auc = knnclassify(traindata[:,selectedFeatures], traindata[:,class_index], testdata[:,selectedFeatures], testdata[:,class_index], 5, weights='uniform')
print("acc_score:", acc_score)
print("auc:", auc)
original_discrimination = calculate_original_discrimination(class_label=y_predict, discrimination_column=data_array_test_copy[:,9])
print("selectedFeatures  all feature  discrimination:", original_discrimination)


y_predict, probility, acc_score, auc = knnclassify(traindata[:,selectedFeatures], traindata[:,class_index], testdata[:,selectedFeatures], testdata[:,class_index], 5, weights='uniform')
print("acc_score:", acc_score)
print("auc:", auc)






# selectedFeatures = [ 0  ,4,   10, 11]
# print("selectedFeatures  no discrimination :", selectedFeatures)
# # selectedFeatures = [i for i in range(class_index)]
# y_predict, probility, acc_score, auc = knnclassify(traindata[:,selectedFeatures], traindata[:,class_index], testdata[:,selectedFeatures], testdata[:,class_index], 5, weights='uniform')
# print("acc_score:", acc_score)
# print("auc:", auc)

