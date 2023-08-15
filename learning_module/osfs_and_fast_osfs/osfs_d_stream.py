


import datetime
from correlation_measure.chi_square_g2_test.my_cond_indep_chisquare import my_cond_indep_chisquare
from learning_module.osfs_and_fast_osfs.computer_dep_2 import computer_dep_2
import numpy as np
def osfs_d(data1,class_index,alpha,test, max_k=3, discrete=1):
    # % for continue value
    #
    n,p=data1.shape
    print("shape:",n,p)
    ns=data1.max(axis=0)
    selected_features=[]
    selected_features1=[]
    b=[]

    start = datetime.datetime.now()

    for i in range(0,p-1):
        #     %for very sparse data
        n1=sum(data1[:,i])
        if n1==0:
            continue

        stop=0
        CI=1

        CI, dep, alpha2 = my_cond_indep_chisquare(data1,i, class_index, [], test, alpha, ns)

        if CI==0:
             stop=1
             if isinstance(selected_features,np.ndarray):
                 selected_features = selected_features.tolist()
             selected_features.append(i)

        if stop:
           p2=len(selected_features)
           selected_features1=selected_features

           for j in range(p2):
               b=np.setdiff1d(selected_features1, selected_features[j])

               if b.tolist() != []:
                    CI,dep1,p_value =computer_dep_2(b,selected_features[j],class_index,max_k, discrete, alpha, test,data1)
                    if CI==1:
                         selected_features1=b

        selected_features=selected_features1
# end
#
#  time=toc(start);
    time = datetime.datetime.now() - start

    return selected_features, time




#function   [selected_features, time]=osfs_d(data1,class_index,alpha,test)
#% for continue value
#
#[n,p]=size(data1);
#ns=max(data1);
#selected_features=[];
#selected_features1=[];
#b=[];
#
#start=tic;
#
# for i=1:p-1
#
#
#     %for very sparse data
#     n1=sum(data1(:,i));
#      if n1==0
#        continue;
#      end
#
#
#     stop=0;
#     CI=1;
#
#     [CI] = my_cond_indep_chisquare(data1,i, class_index, [], test, alpha, ns);
#
#      if CI==0
#          stop=1;
#          selected_features=[selected_features,i];
#      end
#
#      if stop
#
#          p2=length(selected_features);
#          selected_features1=selected_features;
#
#           for j=1:p2
#
#              b=setdiff(selected_features1, selected_features(j),'stable');
#
#               if ~isempty(b)
#                  [CI]=compter_dep_2(b,selected_features(j),class_index,3, 1, alpha, test,data1);
#
#                   if CI==1
#                      selected_features1=b;
#                   end
#              end
#          end
#     end
#   selected_features=selected_features1;
# end
#
#  time=toc(start);