import re
from tqdm import tqdm
import pickle
from trie import *
from lev import *
from error_model import *
from word_candidates import *
from sentence_candidates import *
from language_model import *
from utils import *
from splits import *
from classifier import *
from iterations import *
import sys
import time

if __name__ == '__main__':
    clean_tree = build_super_clean_tree()
    print("built clean tree model", file=sys.stderr)
    clean_tree_rever = build_super_clean_tree_rever()
    print("built reversed clean tree model", file=sys.stderr)
    error_model_rever = build_error_model_rever()
    print("built reversed error model", file=sys.stderr)
    error_model = build_error_model()
    print("built error model", file=sys.stderr)
    clean_unigrams, clean_language_model = build_clean_language_model()
    print("loaded language model", file=sys.stderr)
    print("loading model succeeded", file=sys.stderr)
    print(time.ctime(), file=sys.stderr)
    english = "qwertyuiop[]asdfghjkl;'zxcvbnm,.`" + 'QWERTYUIOP{}ASDFGHJKL:"ZXCVBNM<>?~'
    russian = "йцукенгшщзхъфывапролджэячсмитьбюёЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮ,Ё"
    rus_to_eng = str.maketrans(russian, english)
    eng_to_rus = str.maketrans(english, russian) 
    #for line in sys.stdin:
    #while True:
    for line in sys.stdin:
        #line = input()
        tmp = [line]
        сhanges3 = False
        tokens_with_p = tmp[0].strip('\n').split()
        tokens = preprocess_string(tmp[0])
        if tokens == []:
            print(tmp[0].strip('\n'))
            continue
        prev, changes3 = keyboard_swapper(clean_unigrams, rus_to_eng, eng_to_rus, tokens_with_p, tokens)
        if changes3 == False:
            spaces = rip(tmp[0].strip('\n'))
        else:
            prev = preprocess_string(' '.join(prev))
        mask = upper_mask(tmp[0])
        prev = preprocessing(prev)
        local_res = iterations(clean_language_model, clean_unigrams, clean_tree, clean_tree_rever, error_model, error_model_rever, prev)
        ans1 = ''
        if len(local_res) == len(mask) and changes3 == False:
            try:
                local_res = tok_corr_upd(tokens, local_res, tokens_with_p, error_model)
                if len(local_res)!=1:
                    for p in range(0, len(local_res)-1):
                        ans1 +=local_res[p]
                        ans1 +=spaces[p]
                    ans1+=local_res[-1]
                    ans1+='\n'
                else:
                    ans1+=local_res[0]
                    ans1+='\n'
            except:
                ans1 = tmp[0]
                print(ans1.strip('\n'))
                continue
        else:
            ans1 = ' '.join(local_res)+'\n'
        print(ans1.strip('\n'))