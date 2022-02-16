import numpy as np

def m_f(a, b):
    if a != b:
        return 1
    return 0

def lev(word1, word2):
    l1 = len(word1)
    l2 = len(word2)
    mat = np.zeros((l2+1, l1+1), dtype = 'int')
    mat[0,:] = [i for i in range(l1+1)]
    mat[:, 0] = [i for i in range(l2+1)]
    for i in range(1, l2+1):
        for j in range(1, l1+1):
            mat[i, j] = min(mat[i-1, j]+1, mat[i, j-1]+1, mat[i-1, j-1]+m_f(word2[i-1], word1[j-1]))
    return mat[l2 , l1]
    
def lev_back_prior(word1, word2):
    l1 = len(word1)
    l2 = len(word2)
    mat = np.zeros((l2+1, l1+1), dtype = 'int')
    mat[0,:] = [i for i in range(l1+1)]
    mat[:, 0] = [i for i in range(l2+1)]
    for i in range(1, l2+1):
        for j in range(1, l1+1):
            mat[i, j] = min(mat[i - 1, j]+1, mat[i, j-1]+1, mat[i-1, j-1]+m_f(word2[i-1], word1[j-1]))
    j = l1
    i = l2
    res = []
    while(True):
        if i > 0 and j > 0:
            left = mat[i, j-1]
            up = mat[i-1, j]
            diag = mat[i-1, j-1]
            min_num = min(left, up, diag)
            if min_num == diag:
                res.append(word2[i-1] + word1[j-1])
                i -= 1
                j -= 1
            elif min_num == up:
                res.append(word2[i-1] + '_')
                i -= 1
            elif min_num == left:
                res.append('_' + word1[j-1])
                j -= 1
        elif i == 0 and j != 0:
            res.append('_' + word1[j-1])
            j -= 1
        elif i != 0 and j == 0:
            res.append(word2[i-1] + '_')
            i -= 1
        else:
            break
    return res