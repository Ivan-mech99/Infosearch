from utils import *
from tqdm import tqdm
import numpy as np
import pickle

def build_clean_language_model(name = 'queries_all.txt', encoding = 'utf-8', a = 0.001, b = 0.001, const_f = -15):
    bgrams = dict()
    unigrams = dict()
    total_words = 0
    with open(name, encoding = encoding) as file:    
        for count, line in enumerate(file):
            tmp = line.split('\t')
            if len(tmp) == 2:
                orig = preprocess_string(tmp[1])
            else:
                orig = preprocess_string(tmp[0])
            if orig != []:
                for elem in orig:
                    total_words +=1
                    if elem not in unigrams:
                        unigrams[elem] = 1
                    else:
                        unigrams[elem] += 1
                data = list(zip(orig, orig[1:]))
                for elem in data:
                    if elem[0] not in bgrams:
                        bgrams[elem[0]] = dict()
                    if elem[1] not in bgrams[elem[0]]:
                        bgrams[elem[0]][elem[1]] = 1
                    else:
                        bgrams[elem[0]][elem[1]] += 1
    num_uniq = len(unigrams)
    add_coeff = a * num_uniq
    for key in unigrams:
        unigrams[key] = np.log((unigrams[key] + a) / (total_words + add_coeff))
    unigrams['<non>'] = np.log(a / (total_words + add_coeff))
    for key1 in bgrams:
        loc_total = 0
        for key2 in bgrams[key1]:
            loc_total+=bgrams[key1][key2]
        add_coeff = b * loc_total
        for key2 in bgrams[key1]:
            bgrams[key1][key2] = np.log((bgrams[key1][key2] + b)/(add_coeff + loc_total))
        if const_f is not None:
            bgrams[key1]['<non>'] = const_f
        else:
            bgrams[key1]['<non>'] = np.log(b/(add_coeff + loc_total))
    #with open("clean_bgrams.pickle", "wb") as fw:
    #    pickle.dump(bgrams, fw)
    #with open("clean_unigrams.pickle", "wb") as fw:
    #    pickle.dump(unigrams, fw)
    return unigrams, bgrams