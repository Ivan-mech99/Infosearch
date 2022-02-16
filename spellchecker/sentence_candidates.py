import re
from tqdm import tqdm
from trie import *
from lev import *
from error_model import *
from word_candidates import *

class Scand:
    def __init__(self, prefix='<eos>', pos_tree = 0, wt = 0, ans = list()):
        self.pos_tree = pos_tree
        self.prefix = prefix
        self.ans = []
        self.ans.extend(ans)
        self.wt = wt
    def __repr__(self):
        return str("pos_tree: "+str(self.pos_tree)+' '+str(self.ans)+' '+str(self.wt))
    def __lt__(self, other):
        return self.wt < other.wt

def graph_gen_var2(tree, tree_rev, error_model, error_model_rev, lang_model, unigrams, tokenized_q, max_sz = 200, max_res_sz = 10, alpha = 0.5, num_leafs=10):
    tmp = []
    for i, word in enumerate(tokenized_q):
        tmp.append(check_one_word_arg_dict(tree, error_model, unigrams, word, max_sz = max_sz, max_res_sz = max_res_sz, alpha = alpha))
        tmp[i].extend(check_one_word_reversed(tree_rev, error_model_rev, error_model, unigrams, word, max_sz = max_sz, max_res_sz = 10, alpha = alpha))
    max_len = 0
    for el in tmp:
        x = len(el)
        if x>max_len:
            max_len = x
    search_tree = Trie()
    lim = len(tmp[0])
    lim2 = len(tokenized_q)-1
    search_tree.nodes_data[0].child_ids = set([1*k for k in range(1, lim+1)])
    for i, el in enumerate(tmp):
        lim = max_len
        if i!=lim2:
            childs = set([1*k+((i+1)*lim) for k in range(1, len(tmp[i+1])+1)])
        else:
            childs = set()
        for j, tok in enumerate(el):
            ind = i*lim+j+1
            search_tree.nodes_data[ind] = Node(ind, tok.prefix, childs)
            search_tree.nodes_data[ind].num = tok.total_wt
            if childs == set():
                search_tree.nodes_data[ind].is_word = True
    return search_tree

def querry_correction1(search_tree, lang_model, mrez=30):
    eps = 1e-14
    max_sz = 30
    cur_sz = 0
    res_sz = 0
    max_res_sz = mrez
    res = []
    myheap = []
    fines_heap = []
    fines_res = []
    cur_obj = Scand(ans=['<eos>'])
    flag = True
    while(flag):
        for elem in search_tree.nodes_data[cur_obj.pos_tree].child_ids:
            if flag == False:
                break
            wt_upd = 0
            tmp = lang_model.get(cur_obj.prefix, None)
            if cur_obj.prefix != '<eos>':
                if tmp is not None:
                    wt_upd += 0.5*lang_model[cur_obj.prefix].get(search_tree.nodes_data[elem].symb, -100)
                else:
                    wt_upd += -100
            wt_upd += (0.5*search_tree.nodes_data[elem].num)
            new_cand = Scand(search_tree.nodes_data[elem].symb, elem, cur_obj.wt+wt_upd)
            new_cand.ans.extend(cur_obj.ans)
            new_cand.ans.append(search_tree.nodes_data[elem].symb)
            if search_tree.nodes_data[elem].is_word == True:
                new_cand.ans.remove('<eos>')
                res.append((new_cand.ans, new_cand.wt))
                fines_res.append(new_cand.wt)
                res_sz+=1
                if res_sz > max_res_sz:
                    res_sz-=1
                    del_id = np.argmin(fines_res)
                    fines_res.pop(del_id)
                    res.pop(del_id)
            else:
                myheap.append(new_cand)
                fines_heap.append(new_cand.wt)
                cur_sz +=1
                if cur_sz > max_sz:
                    del_id = np.argmin(fines_heap)
                    fines_heap.pop(del_id)
                    myheap.pop(del_id)
                    cur_sz -=1
        if(len(myheap)) == 0:
            break
        del_id = np.argmax(myheap)
        cur_obj = myheap.pop(del_id)
        fines_heap.pop(del_id)
        cur_sz -=1
    return res