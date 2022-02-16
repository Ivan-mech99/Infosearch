import numpy as np

def naive_general_classifier(bgrams, unigrams, sent, const_f = -15, alpha = 0.6, beta = 0.4):
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

def naive_main_classifier(bgrams, unigrams, join, split, fix):
    res = []
    for el in join:
        res.append((el[0], naive_general_classifier(bgrams, unigrams, el[0]), True))
    for el in split:
        res.append((el[0], naive_general_classifier(bgrams, unigrams, el[0]), True))
    for el in fix:
        res.append((el[0], naive_general_classifier(bgrams, unigrams, el[0]), False))
    res.sort(key = lambda x: x[1], reverse = True)
    return res[0][0], res[0][2]