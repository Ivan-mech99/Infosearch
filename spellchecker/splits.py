import time
import numpy as np

def one_join_gener(bgrams, unigrams, sent, lim=1):
    sent = ' '.join(sent)
    sent = sent.strip()
    spaces = [i for i in range(len(sent)) if sent[i]==' ']
    variants = [[sent[:i]+sent[1+i:]] for i in spaces]
    res = []
    for var in variants:
        var1 = var[0].split()
        res.append((var1, naive_split_join_classifier(bgrams, unigrams, var1), True))
    res.sort(key=lambda x: x[1], reverse = True)
    return res[:lim]

def one_split_gener(bgrams, unigrams, sent, lim=1):
    sent = ' '.join(sent)
    spaces = [i for i in range(1,len(sent)) if sent[i]!=' ']
    variants = [[sent[:i]+' '+sent[i:]] for i in spaces]
    res = []
    for var in variants:
        var1 = var[0].split()
        res.append((var1, naive_split_join_classifier(bgrams, unigrams, var1), True))
    res.sort(key=lambda x: x[1], reverse = True)
    return res[:lim]

def naive_split_join_classifier(bgrams, unigrams, sent, const_f = -15, alpha = 0.7, beta = 0.3):
    pairs = list(zip(sent[::], sent[1::]))
    def_dict = {'<non>':const_f}
    bg = [bgrams.get(pair[0], def_dict).get(pair[1], const_f) for pair in pairs]
    ug = [unigrams.get(word, unigrams['<non>']) for word in sent]
    if bg==[]:
        bg.append(0)
    if ug==0:
        ug.append(0)
    res = alpha*np.mean(bg)+beta*np.mean(ug)
    return res

def naive_keyboard_clf(unigrams, tokens):
    res = 0
    for word in tokens:
        res += unigrams.get(word, unigrams['<non>'])
    return res

def keyboard_swapper(unigrams, rus_to_eng, eng_to_rus, tokens, prepr):
    res_rus = []
    res_eng = []
    for token in tokens:
        res_rus.append(token.translate(eng_to_rus).lower())
    for token in tokens:
        res_eng.append(token.translate(rus_to_eng).lower())
    res = []
    res.append((res_rus, naive_keyboard_clf(unigrams, res_rus), True))
    res.append((res_eng, naive_keyboard_clf(unigrams, res_eng), True))
    res.append((prepr, naive_keyboard_clf(unigrams, prepr)+0.1, False))
    res.sort(key = lambda x: x[1], reverse = True)
    return res[0][0], res[0][2]

def naive_fix_classifier(res):
    res.sort(key = lambda x: x[1])
    return [res[-1]]