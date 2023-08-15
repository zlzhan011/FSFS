import numpy as np
from correlation_measure.chi_square_g2_test.my_cond_indep_chisquare import my_cond_indep_chisquare
from correlation_measure.fisher_z_test.my_cond_indep_fisher_z import my_cond_indep_fisher_z
def optimal_computer_dep_2(bcf,var,target,max_k, discrete, alpha, test,data):
    """
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #
    #%for a discrete data set, discrete=1, otherwise, discrete=0 for a continue data set
    #
    #%if new feature X is not redundant, that is, we cannot
    #%remove X from X, then we check redundency for each feature.
    #%When we test redundency for a feature, we only consider
    #%its candidate Markov blanlets containg the new feautre X to redunce the
    #%number of tests. for example, now BCF=[2,3,4,5], If feature 6 is added
    #%into BCF, BCF=[2,3,4,5,6]. When testing feature 5, we only consider the
    #%following subsets: [6],[2,6],[3,6],[4,6],[2,3,6],[2,4,6],[3,4,6],if
    #%max_k=3.

    """
    dep1=0
    x=0

    n_pc=len(bcf)
    code=bcf
    N=data.shape[0]
    max_cond_size=max_k
    CI=0;
    p=1;
    if(max_cond_size>n_pc):
        max_cond_size=n_pc

    cond=[];
    cond_size=1;
    while cond_size<=max_cond_size:
          print("cond_size:", cond_size)
          cond_index=np.zeros((1,cond_size))
          for i in range(cond_size):
            cond_index[0][i]=i
            stop=0;

          while stop==0:
                cond=[];
                for i in range(cond_size):
                   if i==(cond_size-1):
                       # print("cond_size:", cond_size)
                       # print("cond_index:", cond_index)
                       # print("n_pc:", n_pc)
                       # print("i:", i)
                       cond_index[0][i]=n_pc-1;
                       cond.append(code[int(cond_index[0][i])])
                   else:
                        cond.append(code[int(cond_index[0][i])])
                print("var:", var, "---target:", target, "---cond:",cond)
                if discrete==1:
                    CI, dep, alpha2 =my_cond_indep_chisquare(data,var,target,cond,test,alpha)
                    x=dep
                else:
                    CI, r, p= my_cond_indep_fisher_z(data,var,target, cond, N, alpha)
                    x=r
                if(CI==1 or np.isnan(x)):
                   stop=1
                   cond_size=max_cond_size+1
                if(stop==0):
                   cond_index,stop=optimal_next_cond_index(n_pc,cond_size,cond_index)
          cond_size=cond_size+1
    dep1=x
    return CI,dep1

def optimal_next_cond_index(n_pc,cond_size,cond_index1):

    stop=1;
    i=cond_size

    if not isinstance(cond_index1,list):
        cond_index1 = cond_index1.tolist()

    cond_index1 = [item + 1 for item in cond_index1[0]]
    cond_index1 = ['NAN'] + cond_index1

    while i>=1:
        if(cond_index1[i]<(n_pc+i-cond_size)):
            if i==cond_size:
                cond_index1[i]=n_pc+i-cond_size;
            else:
                cond_index1[i]=cond_index1[i]+1

            if i<cond_size:
                for j in range(i+1,cond_size):
                    if (j==cond_size):
                        cond_index1[j]=n_pc;
                    else:
                        cond_index1[j]=cond_index1[j-1]+1;
            stop=0
            i=-1
        i=i-1;
    cond_index1.pop(0)
    cond_index1 = [item - 1 for item in cond_index1]
    cond_index = np.array(cond_index1).reshape((1, -1))
    return cond_index,stop

