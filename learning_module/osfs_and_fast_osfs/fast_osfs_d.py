import datetime
from correlation_measure.chi_square_g2_test.my_cond_indep_chisquare import my_cond_indep_chisquare
from learning_module.osfs_and_fast_osfs.computer_dep_2 import computer_dep_2
from learning_module.osfs_and_fast_osfs.optimal_compter_dep_2 import optimal_computer_dep_2
import numpy as np
def  fast_osfs_d(data1,class_index,alpha,test,max_k = 3, discrete = 1):
    """

       #
    #
    # %for disccrete data
    #
    #
    # %important note:
    # %for discrete dataset: the feature values of X must be from 1 to max_value(X), for example,feature X has to take consecutive integer values starting from 1,
    # %that is, 1..max_value(X). For example, if max_value(X)=3, this means that feature X takes values {1,2,3}.
    #
    #
    # %input parameter:
    #
    # %data1: data with all features including the class attribute
    # %target: the index of the class attribute ( we assume the class attribute is the last colomn of a data set)
    # %alpha: significant level( 0.01 or 0.05 )
    # %for discrete data: test = 'chi2' for Pearson's chi2 test;'g2' for G2 likelihood ratio test
    #
    # %for example: The UCI dataset wdbc with 569 instances and 31 features (the index of the  class attribute is 31).
    #
    # %[selected_features1,time]=fast_osfs_d(wdbc,31,0.01,'g2')
    #
    # %if the feature values of X must be from 0 to max_value(X)-1,then
    #
    # %[selected_features1,time]=fast_osfs_d(wdbc+1,31,0.01,'g2')
    #
    # %output:
    # %selected_features1: the selected features
    # %time: running time
    #
    # %please refer to the following papers for the details and cite them:
    # %Wu, Xindong, Kui Yu, Wei Ding, Hao Wang, and Xingquan Zhu. "Online feature selection with streaming features." Pattern Analysis and Machine Intelligence, IEEE Transactions on 35, no. 5 (2013): 1178-1192.
    #
    #
    """


    start = datetime.datetime.now()

    n,p=data1.shape
    ns = data1.max(axis=0)
    selected_features=[]
    selected_features1=[]
    b=[]

    for i in range(p-1): #%the last feature is the class attribute, i.e., the target)

    #     %for very sparse data
         n1=sum(data1[:,i])
         if n1==0:
           continue

         stop = 0
         CI = 1
         CI, dep, alpha2 = my_cond_indep_chisquare(data1, i, class_index, [], test, alpha, ns)

         if CI==0:
             stop=1

         if stop:
    #
             if selected_features!=[]:
    #                    [CI]=compter_dep_2(selected_features,i,class_index,3, 1, alpha, test,data1);
                CI,dep,p_value=computer_dep_2(selected_features,i,class_index,max_k, discrete, alpha, test,data1);
             if CI==0:

                 if isinstance(selected_features, list):
                     selected_features.append(i)
                 else:
                     selected_features = selected_features.tolist()
                     selected_features.append(i)
                 p2=len(selected_features)
                 selected_features1=selected_features;
    #
                 for j in range(p2):

                    b=np.setdiff1d(selected_features1,selected_features[j])
                    if b.tolist() != []:
                             # [CI]=optimal_compter_dep_2(b,selected_features(j),class_index,3, 1, alpha, test,data1);
                        CI, dep = optimal_computer_dep_2(b, selected_features[j], class_index, max_k, discrete, alpha, test,
                                                              data1);

                        if CI==1:
                           selected_features1=b

         selected_features=selected_features1;
    time = datetime.datetime.now() - start

    return selected_features,time




