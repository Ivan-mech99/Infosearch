import numpy as np
from word_candidates import *
from sentence_candidates import *
from splits import *
from classifier import *

def iterations(clean_language_model, clean_unigrams, clean_tree, clean_tree_rever, error_model, error_model_rever, prev, num_it = 1):
    for i in range(0, num_it):
        joinres = one_join_gener(clean_language_model, clean_unigrams, prev)
        splitres = one_split_gener(clean_language_model, clean_unigrams, prev)
        search_tree = graph_gen_var2(clean_tree, clean_tree_rever, error_model, error_model_rever, clean_language_model, clean_unigrams, prev, max_sz = 40, max_res_sz = 25, alpha = 0.)
        res = querry_correction1(search_tree, clean_language_model, 1)
        local_res = naive_fix_classifier(res)
        local_res, flag4 = naive_main_classifier(clean_language_model, clean_unigrams, joinres, splitres, local_res)
        prev = local_res
    return prev
