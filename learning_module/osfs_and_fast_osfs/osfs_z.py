import datetime
from correlation_measure.fisher_z_test.my_cond_indep_fisher_z import my_cond_indep_fisher_z
from learning_module.osfs_and_fast_osfs.computer_dep_2 import computer_dep_2
import numpy as np


def osfs_z(data1,class_index,alpha,max_k = 3, discrete = 0):
    # % for continue value
    #
    n,p=data1.shape
    ns=data1.max(axis=0)
    selected_features=[];
    selected_features1=[];
    b=[];

    start = datetime.datetime.now()

    for i in range(p-1):
        n1=sum(data1[:,i])
        if n1 == 0:
            continue

        stop=0;
        CI=1;
        # [CI,dep]=my_cond_indep_fisher_z(data1,i,class_index,[],n,alpha)
        CI, dep, p = my_cond_indep_fisher_z(data1, i, class_index, [], n, alpha=alpha, print_flag=False)
        # print("--i:", i, "CI:", CI, "dep:", dep, "p:", p)
        CI_discri, dep_discri, p_discri = my_cond_indep_fisher_z(data1, i, 9, [], n, alpha=0.05, print_flag= False)
        # print("--i:",i,"CI_discri:", CI_discri, "dep_discri:", dep_discri, "p_discri:", p_discri)
        # print("\n\n")
        if CI == 1 or np.isnan(dep):
            continue

        if CI==0 and CI_discri==1:
            # if CI_discri==1:
            #     print(" -------- stress ------")
            #     print("--i:", i, "CI_discri:", CI_discri, "dep_discri:", dep_discri, "p_discri:", p_discri)
            #     print(" -------- stress ------")

            stop = 1
            if isinstance(selected_features, np.ndarray):
                selected_features = selected_features.tolist()
            selected_features.append(i)

        if stop:
            p2 = len(selected_features)
            selected_features1 = selected_features
    #
            for j in range(p2):

                 b=np.setdiff1d(selected_features1, selected_features[j])

                 if  b.tolist() != []:
                     # CI, dep1, p_value=computer_dep_2(b,selected_features[j],class_index,max_k, discrete, alpha, 'z',data1);
                     test = 'g2'
                     test = 'chi2'

                     CI, dep1, p_value = computer_dep_2(b, selected_features[j], class_index, max_k, discrete, alpha,
                                                        test, data1);


                     if CI==1 or np.isnan(dep):
                         selected_features1=b

        selected_features=selected_features1

    time = datetime.datetime.now() - start
    return selected_features, time


