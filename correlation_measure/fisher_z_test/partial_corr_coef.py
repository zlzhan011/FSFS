import numpy as np

def  partial_corr_coef(S, i, j, Y):
    """
    #% PARTIAL_CORR_COEF Compute a partial correlation coefficient
    #% [r, c] = partial_corr_coef(S, i, j, Y)
    #%
    #% S is the covariance (or correlation) matrix for X, Y, Z
    #% where X=[i j], Y is conditioned on, and Z is marginalized out.
    #% Let S2 = Cov[X | Y] be the partial covariance matrix.
    #% Then c = S2(i,j) and r = c / sqrt( S2(i,i) * S2(j,j) )
    #%
    #
    #% Example: Anderson (1984) p129
    #% S = [1.0 0.8 -0.4;
    #%     0.8 1.0 -0.56;
    #%     -0.4 -0.56 1.0];
    #% r(1,3 | 2) = 0.0966
    #%
    #% Example: Van de Geer (1971) p111
    #%S = [1     0.453 0.322;
    #%     0.453 1.0   0.596;
    #%     0.322 0.596 1];
    #% r(2,3 | 1) = 0.533
    """


    X = [i,j]
    i2 = 1; #% find_equiv_posns(i, X);
    j2 = 2; #% find_equiv_posns(j, X);

    S_X_X = get_array(S, X,X)
    S_X_Y = get_array(S, X, Y)
    S_Y_Y = get_array(S, Y, Y)
    S_Y_X = get_array(S, Y, X)
    # print("-----------start------------------")
    # print("S_X_X:", S_X_X)
    # print("S_X_Y:", S_X_Y)
    # print("S_Y_X:", S_Y_X)
    # print("S_Y_Y:", S_Y_Y)

    S2 = S_X_X - np.dot(np.dot(S_X_Y,np.linalg.inv(S_Y_Y)),S_Y_X)
    c = S2[i2-1,j2-1]
    r = c / np.sqrt(np.dot(S2[i2-1,i2-1] , S2[j2-1,j2-1]))

    return r, c



def get_array(s, index_list_0,index_list_1):
    if index_list_0 == []:
        index_list_0 = [0]
    if index_list_1 == []:
        index_list_1 = [0]
    index_list_0_min = min(index_list_0)
    if index_list_0_min >= 1:
        index_list_0_min = index_list_0_min - 1
    index_list_1_min = min(index_list_1)
    if index_list_1_min >= 1:
        index_list_1_min = index_list_1_min - 1
    return s[index_list_0_min:max(index_list_0),index_list_1_min:max(index_list_1)]

