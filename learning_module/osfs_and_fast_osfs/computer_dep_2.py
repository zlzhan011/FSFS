
import numpy as np
from correlation_measure.chi_square_g2_test.my_cond_indep_chisquare import my_cond_indep_chisquare
from correlation_measure.fisher_z_test.my_cond_indep_fisher_z import my_cond_indep_fisher_z
def computer_dep_2(bcf,var,target,max_k, discrete, alpha, test,data):
    """
    #%for a discrete data set, discrete=1, otherwise, discrete=0 for a continue data set
    #%test = 'chi2' for Pearson's chi2 test'g2' for G2 likelihood ratio test (default)
    :param bcf:
    :param var:
    :param target:
    :param max_k:
    :param discrete:
    :param alpha:
    :param test:
    :param data:
    :return:
    """
    # print("computer_dep_2  start  computer_dep_2")
    dep1=0
    x=0
    n_bcf=len(bcf)
    code=bcf
    N=data.shape[0]
    max_cond_size=max_k
    CI=0
    p_value=1
    if(max_cond_size>n_bcf):
        max_cond_size=n_bcf
    cond=[]
    cond_size=1
    while cond_size<=max_cond_size:
          cond_index=np.zeros((1,cond_size))

          for i in range(cond_size):
              cond_index[0][i]=i

          stop=0
          while stop==0:
                 cond=[]
                 for i in range(cond_size):
                     # if isinstance(cond_index, list) or isinstance(code,list):
                     #     print("cond_index:", cond_index)
                     if not isinstance(code,list):
                         code = code.tolist()
                     if not isinstance(cond_index,list):
                         cond_index = cond_index.tolist()

                     cond = np.hstack((cond, code[int(cond_index[0][i])])).tolist()
                     cond = [int(item) for item in cond]

                 if discrete==1:
                    ns=data.max(axis=0)
                    CI, dep, p_value=my_cond_indep_chisquare(data,var,target,cond,test,alpha,ns)
                    x=dep
                 else:
                    CI, r, p_value= my_cond_indep_fisher_z(data,var,target, cond, N, alpha)
                    x=r

                 if(CI==1 or np.isnan(x)):
                    stop=1
                    cond_size=max_cond_size+1

                 if(stop==0):
                 #     pass
                     cond_index,stop=next_cond_index(n_bcf,cond_size,cond_index)

          cond_size=cond_size+1
    dep1=x
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # function [cond_index,stop]=next_cond_index(n_bcf,cond_size,cond_index1)
    #
    #  stop=1
    #  i=cond_size
    #
    # while i>=1
    #  	    if(cond_index1(i)<n_bcf+i-cond_size)
    #		 cond_index1(i)=cond_index1(i)+1
    #		 if(i<cond_size)
    #			for(j=i+1:cond_size)
    #				cond_index1(j)=cond_index1(j-1)+1
    #            end
    #         end
    #		  stop=0
    #		  i=-1
    #        end
    #        i=i-1

    return CI,dep1,p_value


def next_cond_index(n_bcf,cond_size,cond_index1):

    stop=1
    i=cond_size
    if not isinstance(cond_index1,list):
        cond_index1 = cond_index1.tolist()

    cond_index1 = [item + 1 for item in cond_index1[0]]
    cond_index1 = ['NAN'] + cond_index1


    while i>=1:

        if(cond_index1[i]<(n_bcf+i-cond_size)):
             cond_index1[i]=cond_index1[i]+1
             if(i<cond_size):
                for j in (i+1,cond_size):
                   cond_index1[j]=cond_index1[j-1]+1
             stop=0
             i=-1
        i=i-1
    cond_index1.pop(0)
    cond_index1 = [item - 1 for item in cond_index1]
    cond_index = np.array(cond_index1).reshape((1,-1))
    return cond_index,stop
#function [cond_index,stop]=next_cond_index(n_bcf,cond_size,cond_index1)
#
#  stop=1
#  i=cond_size
#
#while i>=1
#  	    if(cond_index1(i)<n_bcf+i-cond_size)
#		 cond_index1(i)=cond_index1(i)+1
#		 if(i<cond_size)
#			for(j=i+1:cond_size)
#				cond_index1(j)=cond_index1(j-1)+1
#            end
#         end
#		  stop=0
#		  i=-1
#        end
#        i=i-1
#end
#cond_index=cond_index1

#
#function [CI,dep1,p_value]=compter_dep_2(bcf,var,target,max_k, discrete, alpha, test,data)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
#%for a discrete data set, discrete=1, otherwise, discrete=0 for a continue data set
#%test = 'chi2' for Pearson's chi2 test'g2' for G2 likelihood ratio test (default)
#
#dep1=0
#x=0
#n_bcf=length(bcf)
#code=bcf
#N=size(data,1)
#max_cond_size=max_k
#CI=0
#p_value=1
#if(max_cond_size>n_bcf)
#	max_cond_size=n_bcf
#end
#cond=[]
#cond_size=1
#while cond_size<=max_cond_size
#
#       cond_index=zeros(1,cond_size)
#       for i=1:cond_size
#		    cond_index(i)=i
#        end
#        stop=0
#
#while stop==0
#
#		     cond=[]
#
#
#           for i=1:cond_size
#			  cond=[cond code(cond_index(i))]
#           end
#
#
#
#            if discrete==1
#                 ns=max(data)
#                 [CI, dep, p_value]=my_cond_indep_chisquare(data,var,target,cond,test,alpha,ns)
#                 x=dep
#            else
#                 [CI, r, p_value]= my_cond_indep_fisher_z(data,var,target, cond, N, alpha)
#                 x=r
#            end
#
#            if(CI==1||isnan(x))
#				 stop=1
#				 cond_size=max_cond_size+1
#            end
#
#		   if(stop==0)
#		 	[cond_index,stop]=next_cond_index(n_bcf,cond_size,cond_index)
#           end
# end
#     cond_size=cond_size+1
#end
#dep1=x
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#function [cond_index,stop]=next_cond_index(n_bcf,cond_size,cond_index1)
#
#  stop=1
#  i=cond_size
#
#while i>=1
#  	    if(cond_index1(i)<n_bcf+i-cond_size)
#		 cond_index1(i)=cond_index1(i)+1
#		 if(i<cond_size)
#			for(j=i+1:cond_size)
#				cond_index1(j)=cond_index1(j-1)+1
#            end
#         end
#		  stop=0
#		  i=-1
#        end
#        i=i-1
#end
#cond_index=cond_index1
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
