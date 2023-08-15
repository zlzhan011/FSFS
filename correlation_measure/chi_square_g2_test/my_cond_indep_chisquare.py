import numpy as np
from correlation_measure.chi_square_g2_test.chisquared_prob import chisquared_prob

def test_switch_process_2(test, nijk, tijk):
    if test == 'chi2':
        tmpijk = nijk - tijk

        # xi, yj = np.nonzero((tijk < 10))[0].tolist();
        xyz_tmp = np.where((tijk < 10))
        xi,yj,zj = xyz_tmp[0].tolist(), xyz_tmp[1].tolist(), xyz_tmp[2].tolist()
        # if xiyj_tmp == []:
        #     xi = []
        #     yj = []
        # else:
        #     xi = xiyj_tmp[0]
        #     yj = xiyj_tmp[1]

        for i in range(len(xi)):
            tmpijk[xi[i], yj[i],zj[i]] = abs(tmpijk[xi[i], yj[i], zj[i]]) - 0.5

        # warning off;
        tmp = (tmpijk ** 2) / tijk
        # warning on;
        tmp[np.nonzero(tmp == float('inf'))[0].tolist()] = 0;

    elif test == 'g2':
        tmp = nijk / tijk
        tmp[np.nonzero((tmp == float('inf')) + (tmp == 0))[0].tolist()] = 1
        tmp[np.nonzero(tmp != tmp)[0].tolist()] = 1
        tmp = 2 * nijk * np.log(tmp)
    else:
        print('unrecognized test')

    return tmp, nijk, tijk

def test_switch_process(test, nij, tij):
    if test == 'chi2':
       tmpij=nij-tij;

       # [xi yj]=np.nonzero(tij<10)[0].tolist()
       # xi, yj = np.nonzero((tij < 10))[0].tolist();
       # xiyj_tmp = np.nonzero((tij < 10))[0].tolist()
       # if xiyj_tmp == []:
       #     xi = []
       #     yj = []
       # else:
       #     xi = xiyj_tmp[0]
       #     yj = xiyj_tmp[1]

       xyz_tmp = np.where((tij < 10))

       xi, yj = xyz_tmp[0].tolist(), xyz_tmp[1].tolist()

       for i in range(0,len(xi)):
          # tmpij[xi(i),yj(i)]=abs(tmpij(xi(i),yj(i)))-0.5
          tmpij[xi[i], yj[i]] = abs(tmpij[xi[i], yj[i]]) - 0.5;


       # warning off;
       # tmp=(tmpij**2)/tij;
       tmp = (tmpij ** 2) / tij
       # warning on;
       # tmp[np.nonzero(tmp==Inf)]=0;
       tmp[np.nonzero(tmp == float('inf'))[0].tolist()] = 0;
       # tmp = []
    elif test == 'g2':
       # warning off;
       tmp=nij/tij
       # warning on;
       tmp[np.nonzero((tmp==float('inf'))+(tmp==0))[0].tolist()]=1;
       tmp[np.nonzero(tmp!=tmp)[0].tolist()]=1;
       tmp=2*nij*np.log(tmp)

    else:
        print('unrecognized test ' ,test)

    return tmp, nij, tij


def my_cond_indep_chisquare(Data,X, Y, S, test=None, alpha=None, ns=None):
    """

    #% COND_INDEP_CHISQUARE Test if X indep Y given Z
    #%                      using either chisquare test or likelihood ratio test G2
    #
    #
    #
    #% the feature values of X must be from 1 to dom(X), for example, if feature X have
    #% two values, in data, the two values are denoted as [1,2].
    #
    #%
    #% [CI Chi2 Prob_Chi2] = cond_indep_chisquare(X, Y, S, Data, test, alpha, node_sizes)
    #%
    #% Input :
    #%       Data is the data matrix, NbVar columns * N rows
    #%       X is the index of variable X in Data matrix
    #%       Y is the index of variable Y in Data matrix
    #%       S are the indexes of variables in set S
    #%       alpha is the significance level (default: 0.05)
    #%       test = 'chi2' for Pearson's chi2 test
    #%		   'g2' for G2 likelihood ratio test (default)
    #%       ns:node_sizes (default: max(Data'))
    #%
    #% Output :
    #%       CI = test result (1=conditional independency, 0=no)
    #%       Chi2 = chi2 value (-1 if not enough data to perform the test --> CI=1)
    """
    test = 'g2' if test==None and alpha==None and ns==None else test
    alpha = 0.05 if (test == None and alpha == None) or (alpha==None and ns==None) or (test == None and ns==None) else alpha
    ns = Data.max(axis=0) if test == None or alpha!=None or ns==None else ns

    # Y = Y-1
    N = Data.shape[0]
    # %N = size(Data,2)
    qi=ns[S]

    tmp=[1] +np.cumprod(qi[0:qi.shape[0]-1]).tolist()

    if S == []:
        qs =[]
    else:
        qs=1+np.dot((qi-1),np.array(tmp).reshape(1,-1).T)

    dep=-1.0
    alpha2=1

    if qs ==[]:
       nij=np.zeros((ns[X],ns[Y]))
       df=np.dot(np.prod(ns[[X,Y]]-1),np.prod(ns[S]))
    else:
    # %   Commented by Mingyi
    # %    nijk=zeros(ns(X),ns(Y),qs);
    # %    tijk=zeros(ns(X),ns(Y),qs);
    # %   Commention ends
    # %   Added by Mingyi
       nijk=np.zeros((ns[X],ns[Y],1))
       tijk=np.zeros((ns[X],ns[Y],1))
    # %   Addition ends
       df=np.dot(np.prod(ns[[X,Y]]-1),qs)

    # % I add it%%%%%%%%
    if(df<=0):
        df=1

    # %if (N<10*df)
    if (N<5*df):

       # % Not enough data to perform the test
       Chi2=-1;
       CI=1;
       print('Not enough data to perform the test: \t\t\t: INDPCY :\tCHI2=%8.2f \t\n',Chi2);

    elif  S==[]:
           for i in range(0,ns[X]):
               for j in range(0,ns[Y]):

                  nij[i,j]=len(np.nonzero((((Data[:,X].reshape(-1,1))==(i+1))&((Data[:,Y].reshape(-1,1))==(j+1))))[0].tolist())

           restr=np.nonzero(nij.sum(axis=0)==0)[0].tolist()
           if restr!= []:
               nij=nij[:,np.nonzero(nij.sum(axis=0))[0].tolist()]
           tij=np.dot(nij.sum(axis=1).reshape(-1,1),nij.sum(axis=0).reshape(1,-1))/N ;

           tmp, nij, tij = test_switch_process(test, nij, tij)
           Chi2=sum(sum(tmp))
           alpha2=1-chisquared_prob(Chi2,df)
           CI=(alpha2>=alpha)
           # %%%%%%%%%%%%%%%% I add it%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
           statistic = Chi2
           if (alpha2 >= alpha):
                dep = (-2.0) - statistic / df
           else:
                dep = 2.0 + statistic / df

            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    else:

        SizeofSSi=1;
        for exemple in range(N):
            i=Data[exemple,X]
            j=Data[exemple,Y]
            Si=Data[exemple,S]-1
            # print("exemple:", exemple)
            # %Added by Mingyi
            if exemple==0:
               SSi = np.zeros((SizeofSSi,Si.shape[-1]))
               SSi[SizeofSSi-1,:]=Si
               # nijk[i-1,j-1,SizeofSSi-1]=1
               nijk = three_dimensional_array_set_value(nijk, i - 1, j - 1, SizeofSSi - 1,
                                                        1)
            else:
               flag=0
               for iii in range(SizeofSSi):


                   if all(SSi[iii,:]==Si):
                       # nijk[i-1,j-1,SizeofSSi-1]=nijk[i-1,j-1, SizeofSSi-1]+1;
                       nijk = three_dimensional_array_set_value(nijk, i - 1, j - 1, iii,
                                                                nijk[i-1,j-1, iii]+1)

                       flag=1;
        #                end
        #            end
               if flag==0:
                   SizeofSSi=SizeofSSi+1
                   # SSi[SizeofSSi-1,:]=Si

                   SSi = SSi_array_append(SSi, Si)
                   # nijk[i-1,j-1,SizeofSSi-1]=1
                   nijk = three_dimensional_array_set_value(nijk, i-1, j-1, SizeofSSi-1,
                                                     1)

            # print("example:",exemple)
            # print("nijk:", nijk)
            # print("*************")
        #        %Addition ends
        #        %Commented by Mingyi
        # %         k=1+Si*tmp';
        # %         nijk(i,j,k)=nijk(i,j,k)+1;
        #        %Commention ends
        #    end
        #
        first_dim,second_dim,third_dim = nijk.shape
        nik=nijk.sum(axis=1).reshape((first_dim,1,third_dim))
        njk=nijk.sum(axis=0).reshape((1,second_dim,third_dim))
        N2=njk.sum(axis=1).reshape((1,1,third_dim))
        #
        # %   for k=1:qs,         %Commented by Mingyi
        for k in range(SizeofSSi):    #%Added by Mingyi
           if N2[:,:,k].tolist()[0][0]==0:
               tijk[:,:,k]=0
           else:
               set_dim = np.dot(nik[:,:,k], njk[:,:,k])/N2[:,:,k]
               # tijk[:,:,k]=np.dot(nik[:,:,k], njk[:,:,k])/N2[:,:,k]
               tijk = three_dimensional_array_set_dim(tijk, input_first_dim=":", input_second_dim=":", input_third_dim=k,
                                               set_dim=set_dim)

        #        end
        #    end
        #
        tmp, nijk, tijk = test_switch_process_2(test, nijk, tijk)
        #
        Chi2=tmp.sum(axis=0).sum(axis=0).sum(axis=0);
        alpha2=1-chisquared_prob(Chi2,df);
        CI=(alpha2>=alpha)
        #
        # %%%%%%%%%%%%%%%% I add it%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        statistic=Chi2
        if(alpha2>=alpha):
              dep=(-2.0)-statistic/df
        else:
              dep=2.0+statistic/df

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # end
        # %fprintf('\t\t\t: INDPCY :\tCHI2=%8.2f \t\n',Chi2);

    return CI, dep, alpha2



def SSi_array_append(SSi,Si):
    SSi = SSi.tolist()
    SSi.append(Si.tolist())
    # for i in range(len(SSi)):
    #     SSi[i] = SSi[i]  + Si.tolist()
    return np.array(SSi)


def three_dimensional_array_set_value(nijk,input_first_dim, input_second_dim, input_third_dim, set_value):
    first_dim, second_dim, third_dim = nijk.shape
    if third_dim<(input_third_dim+1):
        new_third_dim = np.zeros((first_dim,second_dim,1))
        nijk = np.dstack((nijk, new_third_dim))
        nijk[input_first_dim, input_second_dim, input_third_dim] = set_value
    else:
        nijk[input_first_dim,input_second_dim,input_third_dim] = set_value

    return nijk


def three_dimensional_array_set_dim(nijk,input_first_dim=":", input_second_dim=":", input_third_dim=0, set_dim=[]):
    first_dim, second_dim, third_dim = nijk.shape
    if third_dim<(input_third_dim+1):
        # new_third_dim = np.zeros((first_dim,second_dim,1))
        new_third_dim = set_dim
        nijk = np.dstack((nijk, new_third_dim))
        # nijk[input_first_dim, input_second_dim, input_third_dim] = set_value
    else:
        nijk[:,:,input_third_dim] = set_dim

    return nijk











#function [CI, dep, alpha2] = my_cond_indep_chisquare(Data,X, Y, S, test, alpha, ns)
#
#% COND_INDEP_CHISQUARE Test if X indep Y given Z
#%                      using either chisquare test or likelihood ratio test G2
#
#
#
#% the feature values of X must be from 1 to dom(X), for example, if feature X have
#% two values, in data, the two values are denoted as [1,2].
#
#%
#% [CI Chi2 Prob_Chi2] = cond_indep_chisquare(X, Y, S, Data, test, alpha, node_sizes)
#%
#% Input :
#%       Data is the data matrix, NbVar columns * N rows
#%       X is the index of variable X in Data matrix
#%       Y is the index of variable Y in Data matrix
#%       S are the indexes of variables in set S
#%       alpha is the significance level (default: 0.05)
#%       test = 'chi2' for Pearson's chi2 test
#%		   'g2' for G2 likelihood ratio test (default)
#%       ns:node_sizes (default: max(Data'))
#%
#% Output :
#%       CI = test result (1=conditional independency, 0=no)
#%       Chi2 = chi2 value (-1 if not enough data to perform the test --> CI=1)
#%
#%
#% V1.4 : 24 july 2003 (Ph. Leray - philippe.leray@univ-nantes.fr)
#%
#%
#% Things to do :
#% - do not use 'find' in nij computation (when S=empty set)
#% - find a better way than 'warning off/on' in tmpij, tmpijk computation
#%
#
#if nargin < 5, test = 'g2'; end
#if nargin < 6, alpha = 0.05; end
#if nargin < 7, ns = max(Data); end
#
#N = size(Data,1);
#%N = size(Data,2);
#qi=ns(S);
#tmp=[1 cumprod(qi(1:end-1))];
#qs=1+(qi-1)*tmp';
#
#dep=-1.0;
#alpha2=1;
#
#if isempty(qs),
#    nij=zeros(ns(X),ns(Y));
#    df=prod(ns([X Y])-1)*prod(ns(S));
#else
#%   Commented by Mingyi
#%    nijk=zeros(ns(X),ns(Y),qs);
#%    tijk=zeros(ns(X),ns(Y),qs);
#%   Commention ends
#%   Added by Mingyi
#    nijk=zeros(ns(X),ns(Y),1);
#    tijk=zeros(ns(X),ns(Y),1);
#%   Addition ends
#    df=prod(ns([X Y])-1)*qs;
#end
#
#
#% I add it%%%%%%%%
#if(df<=0)
#	df=1;
#end
#%%%%%%%%%%%%%%%%%%
#
#%if (N<10*df)
# if (N<5*df)
#
#    % Not enough data to perform the test
#    Chi2=-1;
#    CI=1;
#  fprintf('Not enough data to perform the test: \t\t\t: INDPCY :\tCHI2=%8.2f \t\n',Chi2);
#
#elseif isempty(S)
#
#    for i=1:ns(X),
#        for j=1:ns(Y),
#
#
#
#
#           nij(i,j)=length(find(((Data(:,X))==i)&((Data(:,Y))==j))) ;
#        end
#    end
#    restr=find(sum(nij,1)==0);
#    if ~isempty(restr)
#        nij=nij(:,find(sum(nij,1)));
#    end
#
#    tij=sum(nij,2)*sum(nij,1)/N ;
#
# switch test
#    case 'chi2',
#        tmpij=nij-tij;
#
#        [xi yj]=find(tij<10);
#        for i=1:length(xi),
#           tmpij(xi(i),yj(i))=abs(tmpij(xi(i),yj(i)))-0.5;
#        end
#
#        warning off;
#        tmp=(tmpij.^2)./tij;
#        warning on;
#        tmp(find(tmp==Inf))=0;
#
#    case 'g2',
#        warning off;
#        tmp=nij./tij;
#        warning on;
#        tmp(find(tmp==Inf | tmp==0))=1;
#        tmp(find(tmp~=tmp))=1;
#        tmp=2*nij.*log(tmp);
#
#    otherwise,
#        error(['unrecognized test ' test]);
#    end
#
#    Chi2=sum(sum(tmp));
#    alpha2=1-chisquared_prob(Chi2,df);
#    CI=(alpha2>=alpha) ;
#%%%%%%%%%%%%%%%% I add it%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#statistic=Chi2;
#if(alpha2>=alpha)
#       dep=(-2.0)-statistic/df;
#else
#       dep=2.0+statistic/df;
#
#end
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
#else
#    SizeofSSi=1;
#    for exemple=1:N,
#
#
#        i=Data(exemple,X);
#        j=Data(exemple,Y);
#        Si=Data(exemple,S)-1;
#
#
#
#
#
#
#
#        %Added by Mingyi
#        if exemple==1
#            SSi(SizeofSSi,:)=Si;
#            nijk(i,j,SizeofSSi)=1;
#        else
#            flag=0;
#            for iii=1:SizeofSSi
#                if isequal(SSi(iii,:),Si)
#                    nijk(i,j,iii)=nijk(i,j,iii)+1;
#                    flag=1;
#                end
#            end
#            if flag==0
#                SizeofSSi=SizeofSSi+1;
#                SSi(SizeofSSi,:)=Si;
#                nijk(i,j,SizeofSSi)=1;
#            end
#        end
#        %Addition ends
#        %Commented by Mingyi
#%         k=1+Si*tmp';
#%         nijk(i,j,k)=nijk(i,j,k)+1;
#        %Commention ends
#    end
#
#    nik=sum(nijk,2);
#    njk=sum(nijk,1);
#    N2=sum(njk);
#
# %   for k=1:qs,         %Commented by Mingyi
#    for k=1:SizeofSSi    %Added by Mingyi
#        if N2(:,:,k)==0
#            tijk(:,:,k)=0;
#        else
#            tijk(:,:,k)=nik(:,:,k)*njk(:,:,k)/N2(:,:,k);
#        end
#    end
#
#    switch test
#    case 'chi2',
#        tmpijk=nijk-tijk;
#
#        [xi yj]=find(tijk<10);
#        for i=1:length(xi),
#            tmpijk(xi(i),yj(i))=abs(tmpijk(xi(i),yj(i)))-0.5;
#        end
#
#        warning off;
#        tmp=(tmpijk.^2)./tijk;
#        warning on;
#        tmp(find(tmp==Inf))=0;
#
#    case 'g2',
#        warning off;
#        tmp=nijk./tijk;
#        warning on;
#        tmp(find(tmp==Inf | tmp==0))=1;
#        tmp(find(tmp~=tmp))=1;
#        tmp=2*nijk.*log(tmp);
#
#    otherwise,
#        error(['unrecognized test ' test]);
#    end
#
#    Chi2=sum(sum(sum(tmp)));
#    alpha2=1-chisquared_prob(Chi2,df);
#    CI=(alpha2>=alpha) ;
#
#%%%%%%%%%%%%%%%% I add it%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#statistic=Chi2;
#if(alpha2>=alpha)
#       dep=(-2.0)-statistic/df;
#else
#       dep=2.0+statistic/df;
#
#end
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#end
# %fprintf('\t\t\t: INDPCY :\tCHI2=%8.2f \t\n',Chi2);
#
#clear tijk
#clear nijk
#clear nij
#clear tij
#clear tmpijk