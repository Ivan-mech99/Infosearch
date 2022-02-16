import re
from trie import *
from lev import *
from error_model import *
import time

class Cand:
    def __init__(self, orig, prefix='', pos_tree = 0, pos_orig = 0, pref_wt = 0, total_wt = 0):
        self.pos_tree = pos_tree
        self.pos_orig = pos_orig
        self.orig = orig
        self.prefix = prefix
        self.pref_wt = pref_wt
        self.total_wt = total_wt
    def __repr__(self):
        if self.prefix == '':
            return str(str(self.pos_tree)+' '+str(self.pos_orig)+' '+'<eos>'+' '+str(self.total_wt))
        else:
            return str("pos_tree: "+str(self.pos_tree)+' pos_orig:'+str(self.pos_orig)+' '+self.prefix+' '+str(self.total_wt))
    def __lt__(self, other):
        return self.total_wt < other.total_wt
        
def make_cand(tree, error_model, cur_obj, new_pos_tree, alpha = 1, eps = 1e-14):
    if new_pos_tree < 0:
        tmp = abs(new_pos_tree)
        cur_pref = cur_obj.prefix + tree.nodes_data[tmp].symb
        token = '_'+tree.nodes_data[tmp].symb
        pref_wt = cur_obj.pref_wt + error_model.get(token, np.log(eps))
        total_wt = pref_wt + alpha * tree.nodes_data[tmp].num
        res = Cand(cur_obj.orig, cur_pref, tmp, cur_obj.pos_orig, pref_wt, total_wt)
        return res
    elif new_pos_tree == cur_obj.pos_tree and cur_obj.pos_orig < len(cur_obj.orig):
        token = cur_obj.orig[cur_obj.pos_orig] +'_'
        pref_wt = cur_obj.pref_wt + error_model.get(token, np.log(eps))
        total_wt = pref_wt + alpha * tree.nodes_data[cur_obj.pos_tree].num
        res = Cand(cur_obj.orig, cur_obj.prefix, cur_obj.pos_tree, cur_obj.pos_orig+1, pref_wt, total_wt)
        return res
    elif new_pos_tree != cur_obj.pos_tree and cur_obj.pos_orig < len(cur_obj.orig) and new_pos_tree>=0:
        cur_pref = cur_obj.prefix + tree.nodes_data[new_pos_tree].symb
        token = cur_obj.orig[cur_obj.pos_orig]+tree.nodes_data[new_pos_tree].symb
        if token[0] != token[1]:
            pref_wt = cur_obj.pref_wt + error_model.get(token, np.log(eps))
        else:
            pref_wt = cur_obj.pref_wt
        total_wt = pref_wt + alpha * tree.nodes_data[new_pos_tree].num
        res = Cand(cur_obj.orig, cur_pref, new_pos_tree, cur_obj.pos_orig+1, pref_wt, total_wt)
        return res

def check_one_word_arg_dict(tree, error_model, unigrams, word, max_sz = 30, max_res_sz = 15, alpha = 0.1):
    cur_sz = 0
    res_sz = 0
    res = dict()
    res1 = []
    if len(word) <=2:
        for i in range(0, max_res_sz):
            res1.append(Cand(word, prefix = word, total_wt = np.log(0.9)))
        return res1
    myheap = []
    fines = []
    cur_obj = Cand(word)
    cur_fine = 0
    flag = True
    while(flag):
        for elem in list(tree.nodes_data[cur_obj.pos_tree].child_ids):
            if flag == False:
                break
            new_cand = make_cand(tree, error_model, cur_obj, elem, alpha)
            if new_cand is not None:
                if tree.nodes_data[new_cand.pos_tree].is_word == True and new_cand.pos_orig >= len(new_cand.orig):
                    if new_cand.prefix not in res:
                        res[new_cand.prefix] = new_cand.total_wt + unigrams.get(new_cand.prefix, unigrams['<non>'])
                    else:
                        if new_cand.total_wt + tree.nodes_data[new_cand.pos_tree].num > res[new_cand.prefix]:
                            res[new_cand.prefix] = new_cand.total_wt + unigrams.get(new_cand.prefix, unigrams['<non>'])
                    myheap.append(new_cand)
                    fines.append(new_cand.total_wt)
                    cur_sz +=1
                    if cur_sz > max_sz:
                        del_id = np.argmin(fines)
                        del1 = fines.pop(del_id)
                        del2 = myheap.pop(del_id)
                        cur_sz -=1
                    res_sz+=1
                    if res_sz == max_res_sz:
                        flag =False
                        break
                else:
                    myheap.append(new_cand)
                    fines.append(new_cand.total_wt)
                    cur_sz +=1
                    if cur_sz > max_sz:
                        del_id = np.argmin(fines)
                        del1 = fines.pop(del_id)
                        del2 = myheap.pop(del_id)
                        cur_sz -=1
        if(len(myheap)) == 0:
            break
        del_id = np.argmax(fines)
        cur_obj = myheap.pop(del_id)
        cur_fine = fines.pop(del_id)
        cur_sz -=1
    res1 = []
    for key in res:
        x = Cand('r', prefix=key, total_wt = res[key])
        res1.append(x)
    return res1

def reweight_2(error_model, unigrams, fix, orig, alpha=1):
    err_toks = lev_back_prior(fix, orig)
    total_wt = 0
    for tok_ in err_toks:
        if tok_[0]!=tok_[1]:
            try:
                total_wt+=error_model[tok_[0]][tok_[1]]
            except:
                total_wt+=np.log(1e-14)
    total_wt+=unigrams.get(fix[::-1], unigrams['<non>'])
    return total_wt

def check_one_word_reversed(tree, error_model, error_model_norm, unigrams, word, max_sz = 30, max_res_sz = 15, alpha = 0.1):
    cur_sz = 0
    res_sz = 0
    res = dict()
    res1 = []
    if len(word) <= 2:
        for i in range(0, max_res_sz):
            res1.append(Cand(word, prefix = word, total_wt = np.log(0.9)))
        return res1
    word = word[::-1]
    myheap = []
    fines = []
    cur_obj = Cand(word)
    cur_fine = 0
    flag = True
    while(flag):
        for elem in list(tree.nodes_data[cur_obj.pos_tree].child_ids):
            if flag == False:
                break
            new_cand = make_cand(tree, error_model, cur_obj, elem, alpha)
            if new_cand is not None:
                if tree.nodes_data[new_cand.pos_tree].is_word == True and new_cand.pos_orig >= len(new_cand.orig):
                    if new_cand.prefix not in res:
                        res[new_cand.prefix] = new_cand.total_wt + unigrams.get(new_cand.prefix[::-1], unigrams['<non>'])
                    else:
                        if new_cand.total_wt + tree.nodes_data[new_cand.pos_tree].num > res[new_cand.prefix]:
                            res[new_cand.prefix] = new_cand.total_wt + unigrams.get(new_cand.prefix[::-1], unigrams['<non>'])
                    myheap.append(new_cand)
                    fines.append(new_cand.total_wt)
                    cur_sz +=1
                    if cur_sz > max_sz:
                        del_id = np.argmin(fines)
                        del1 = fines.pop(del_id)
                        del2 = myheap.pop(del_id)
                        cur_sz -=1
                    res_sz+=1
                    if res_sz == max_res_sz:
                        flag =False
                        break
                else:
                    myheap.append(new_cand)
                    fines.append(new_cand.total_wt)
                    cur_sz +=1
                    if cur_sz > max_sz:
                        del_id = np.argmin(fines)
                        del1 = fines.pop(del_id)
                        del2 = myheap.pop(del_id)
                        cur_sz -=1
        if(len(myheap)) == 0:
            break
        del_id = np.argmax(fines)
        cur_obj = myheap.pop(del_id)
        cur_fine = fines.pop(del_id)
        cur_sz -=1
    res1 = []
    for key in res:
        x = Cand('r', prefix=key[::-1], total_wt = res[key])
        res1.append(x)
    res2 = []
    for key in res1:
        x2 = Cand('r', prefix=key.prefix[::-1], total_wt = reweight_2(error_model, unigrams, key.prefix, word, alpha=1))
        res2.append(x2)
    return res1