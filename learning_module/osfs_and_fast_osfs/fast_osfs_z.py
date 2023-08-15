# function   [selected_features,time]=fast_osfs_z(data1,class_index,alpha)
#

import datetime
from correlation_measure.chi_square_g2_test.my_cond_indep_chisquare import my_cond_indep_chisquare
from learning_module.osfs_and_fast_osfs.computer_dep_2 import computer_dep_2
from correlation_measure.fisher_z_test.my_cond_indep_fisher_z import my_cond_indep_fisher_z
from learning_module.osfs_and_fast_osfs.optimal_compter_dep_2 import optimal_computer_dep_2
import numpy as np

def fast_osfs_z(data1,class_index,alpha, max_k = 3, discrete = 0):
    """
    #
    # %for continouous data
    #
    # %input parameter:
    #
    # %data1: data with all features including the class attribute.
    # %the class attribute in data matrix has to take consecutive integer values starting from 0 for classification.
    # %target: the index of the class attribute (we assume the class attribute is the last colomn of data1)
    # %alpha: significant level( 0.01 or 0.05 )
    # %for example: The UCI dataset wdbc with 569 instances and 31 features (the index of the class attribute is 31).
    #
    # % [selected_features1,time]=fast_osfs_z(wdbc,31,0.01)
    #
    # %output:
    # %selected_features1: the selected features
    # %time: running time
    #
    # %please refer to the following papers for the details and cite them:
    # %Wu, Xindong, Kui Yu, Wei Ding, Hao Wang, and Xingquan Zhu. "Online feature selection with streaming features." Pattern Analysis and Machine Intelligence, IEEE Transactions on 35, no. 5 (2013): 1178-1192.
    #
    """

    start = datetime.datetime.now()

    n,p=data1.shape

    selected_features=[]
    selected_features1=[]
    b=[]
    for i in range(p-1): #%the last feature is the class attribute, i.e., the target)
#
#
#
        #     %for very sparse data
        n1=sum(data1[:,i])

        if n1 == 0:
            continue
        stop=0
        CI=1
        CI,dep,p=my_cond_indep_fisher_z(data1,i,class_index,[],n,alpha);

        if CI==1 or np.isnan(dep):
           continue

        if CI==0:
           stop=1

        if stop:

            if selected_features!=[]:

                CI,dep,p_value=computer_dep_2(selected_features,i,class_index,max_k, discrete, alpha, 'z',data1)

            if CI==0 and not np.isnan(dep):
                if isinstance(selected_features,list):
                    selected_features.append(i)
                else:
                    selected_features = selected_features.tolist()
                    selected_features.append(i)
                    # selected_features=[selected_features,i]  #%adding i to the set of selected_features
                p2=len(selected_features)
                selected_features1=selected_features
                # print("selected_features1:", selected_features1)
                for j in range(p2):

                    b = np.setdiff1d(selected_features1, selected_features[j])

                    if b.tolist() != []:
                        # print("b:", b, "-----optimal_computer_dep_2-----selected_features[j]:", selected_features[j])
                        CI,dep=optimal_computer_dep_2(b,selected_features[j],class_index,max_k, discrete, alpha, 'z',data1)

                        if CI==1 or np.isnan(dep):
                            selected_features1=b

        selected_features=selected_features1
    time = datetime.datetime.now() - start
    return selected_features,time


