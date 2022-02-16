import re
from tqdm import tqdm
import pickle
import numpy as np
from utils import *

class Node:
    def __init__(self, uniq_id, symb, child_ids, is_word=False):
        self.uniq_id = uniq_id
        self.symb = symb
        self.child_ids = set()
        self.num = 0
        self.is_word = is_word
        for elem in child_ids:
            self.child_ids.add(elem)
    def _upd(self):
        self.num+=1
    def __repr__(self):
        return str("id: "+str(self.uniq_id)+' val: '+'\''+self.symb+'\''+ ' num: '+ 
                   str(self.num)+' is_word: '+str(self.is_word)+' children: '+str(self.child_ids))
                   
class Trie:
    def __init__(self):
        self.nodes_data = dict()
        self.nodes_data[0] = Node(0, '', [])
        self.max_id = 0
    def add(self, word):
        cur_pos = 0
        self.nodes_data[0].num+=1
        for symb in word:
            found = False
            for child in self.nodes_data[cur_pos].child_ids:
                if self.nodes_data[child].symb == symb:
                    cur_pos = child
                    self.nodes_data[cur_pos]._upd()
                    found = True
                    break
            if found == False:
                self.max_id += 1
                self.nodes_data[self.max_id] = Node(self.max_id, symb, {})
                self.nodes_data[cur_pos].child_ids.add(self.max_id)
                cur_pos = self.max_id
                self.nodes_data[cur_pos]._upd()
        self.nodes_data[cur_pos].is_word = True
    def __repr__(self):
        res = ''
        for key in self.nodes_data:
            print(self.nodes_data[key])
        return ''
    
def reforge_tree(tree):
    for key in tree.nodes_data:
        tmp = set()
        for elem in tree.nodes_data[key].child_ids:
            tmp.add(-elem)
        if key != 0:
            tmp.add(key)
        for elem in tmp:
            tree.nodes_data[key].child_ids.add(elem)
    return tree
        
def build_clean_tree(name = 'queries_all.txt', encoding = 'utf-8'):
    az = Trie()
    with open(name, encoding = encoding) as file: 
        for count, line in enumerate(file):
            tmp = line.split('\t')
            if len(tmp) == 2:
                orig = preprocess_string(tmp[1], badlen = 25)
            else:
                orig = preprocess_string(tmp[0], badlen = 25)
            for word in orig:
                az.add(word)
    total = az.nodes_data[0].num
    for key in az.nodes_data:
        az.nodes_data[key].num /= total
        az.nodes_data[key].num = np.log(az.nodes_data[key].num)
    az = reforge_tree(az)
    #with open("clean_tree.pickle", "wb") as fw:
    #    pickle.dump(az, fw)
    return az
        
def build_super_clean_tree(name = 'queries_all.txt', encoding = 'utf-8'):
    az = Trie()
    with open(name, encoding = encoding) as file: 
        for count, line in enumerate(file):
            tmp = line.split('\t')
            if len(tmp) == 2:
                orig = preprocess_string(tmp[1], badlen = 25)
            else:
                continue
            for word in orig:
                az.add(word)
    total = az.nodes_data[0].num
    for key in az.nodes_data:
        az.nodes_data[key].num /= total
        az.nodes_data[key].num = np.log(az.nodes_data[key].num)
    az = reforge_tree(az)
    #with open("clean_tree_sc.pickle", "wb") as fw:
    #    pickle.dump(az, fw)
    return az

def build_super_clean_tree_rever(name = 'queries_all.txt', encoding = 'utf-8'):
    az = Trie()
    with open(name, encoding = encoding) as file: 
        for count, line in enumerate(file):
            tmp = line.split('\t')
            if len(tmp) == 2:
                orig = preprocess_string(tmp[1], badlen = 25)
            else:
                continue
            for word in orig:
                az.add(word[::-1])
    total = az.nodes_data[0].num
    for key in az.nodes_data:
        az.nodes_data[key].num /= total
        az.nodes_data[key].num = np.log(az.nodes_data[key].num)
    az = reforge_tree(az)
    #with open("clean_tree_sc_rev.pickle", "wb") as fw:
    #    pickle.dump(az, fw)
    return az