import numpy as np
from lev import *
from tqdm import tqdm
from utils import *
import pickle

def build_error_model(name = 'queries_all.txt', encoding = 'utf-8'):
    err_data = dict()
    total_errors = 0
    with open(file = name, encoding = encoding) as file:    
        for count, line in enumerate(file):
            tmp = line.split('\t')
            if len(tmp) == 2:
                orig = preprocess_string(tmp[0])
                fix = preprocess_string(tmp[1])
                if len(orig) == len(fix) and len(orig) != 0:
                    for ind in range(len(fix)):
                        res = lev_back_prior(fix[ind], orig[ind])
                        for bg in res:
                            if bg[0] != bg[1]:
                                total_errors+=1
                                if bg[0] not in err_data:
                                    err_data[bg[0]] = dict()
                                if bg[1] not in err_data[bg[0]]:
                                    err_data[bg[0]][bg[1]] = 1
                                else:
                                    err_data[bg[0]][bg[1]] += 1
    for key1 in err_data:
        for key2 in err_data[key1]:
            err_data[key1][key2] /= total_errors
            err_data[key1][key2] = np.log(err_data[key1][key2])
    #with open("error_model.pickle", "wb") as fw:
    #    pickle.dump(err_data, fw)
    return err_data
    
def build_error_model_rever(name = 'queries_all.txt', encoding = 'utf-8'):
    err_data = dict()
    total_errors = 0
    with open(file = name, encoding = encoding) as file:    
        for count, line in enumerate(file):
            tmp = line.split('\t')
            if len(tmp) == 2:
                orig = preprocess_string(tmp[0])
                fix = preprocess_string(tmp[1])
                if len(orig) == len(fix) and len(orig) != 0:
                    for ind in range(len(fix)):
                        res = lev_back_prior(fix[ind][::-1], orig[ind][::-1])
                        for bg in res:
                            if bg[0] != bg[1]:
                                total_errors+=1
                                if bg[0] not in err_data:
                                    err_data[bg[0]] = dict()
                                if bg[1] not in err_data[bg[0]]:
                                    err_data[bg[0]][bg[1]] = 1
                                else:
                                    err_data[bg[0]][bg[1]] += 1
    for key1 in err_data:
        for key2 in err_data[key1]:
            err_data[key1][key2] /= total_errors
            err_data[key1][key2] = np.log(err_data[key1][key2])
    #with open("error_model_rev.pickle", "wb") as fw:
    #    pickle.dump(err_data, fw)
    return err_data