import re
import numpy as np

def preprocess_string(q, badlen = 25):
    words = q.lower().split()
    res = []
    words = list(filter(lambda x: x != '', words))
    for word in words:
        tmp = re.sub(r'[^а-яa-zёі]', '', word)
        val = len(tmp)
        if val == 0 or val >= badlen:
            return []
        res.append(tmp)
    return res

def upper_mask(q, badlen = 25):
    words = q.split(' ')
    res = []
    words = list(filter(lambda x: x != '', words))
    for word in words:
        tmp = re.sub(r'[^А-ЯA-Zа-яa-zёі]', '', word)
        val = len(tmp)
        if val == 0 or val >= badlen:
            return []
        res.append(tmp)
    res = [np.array([int(x.isupper()) for x in elem]).sum() for elem in res]
    return res

def rip(sent):
    res = []
    flag = 0
    token = ''
    for letter in sent:
        if (letter!=' ' and letter!='\t') and flag==0:
            continue
        elif (letter==' ' or letter == '\t') and flag ==0:
            token = ''
            token+=letter
            flag =1
        elif flag==1 and (letter==' ' or letter == '\t'):
            token+=letter
        elif flag==1 and (letter!=' ' and letter != '\t'):
            res.append(token)
            token=''
            flag = 0
    return res

def tok_corr_upd(prepr, gener, init, error_model):
    good_symb = set(error_model.keys())
    good_symb.remove('_')
    res = []
    for i, word in enumerate(gener):
        if len(word) == len(prepr[i]):
            res_tok = ''
            cur_len = 0
            for letter in init[i]:
                if letter.lower() in good_symb:
                    if letter.isupper():
                        res_tok+=word[cur_len].upper()
                        cur_len+=1
                    else:
                        res_tok+=word[cur_len]
                        cur_len+=1
                else:
                    res_tok+=letter
            res.append(res_tok)
        else:
            word = creative_punct(init[i].lower(), word, error_model)
            num_cap = 0
            for letter in init[i]:
                if letter.isupper():
                    num_cap+=1
            if num_cap>1:
                res.append(word.upper())
            elif num_cap == 1:
                word1 = ''
                for k, symb in enumerate(word):
                    if k==0:
                        word1+=symb.upper()
                    else:
                        word1+=symb
                res.append(word1)
            else:
                res.append(word)
    return res

def creative_punct(orig, fix, error_model):
    good_symb = set(error_model.keys())
    good_symb.remove('_')
    num1 = 0
    for symb in orig:
        if symb not in good_symb:
            num1+=1
    num2 = num1
    res = ''
    i2 = 0
    j1 = len(orig)
    j2 = len(fix)
    for i in range(0, j1):
        if num1!=0 and i2<j2:
            if orig[i] not in good_symb:
                res+=orig[i]
                num1-=1
            else:
                if orig[i] == fix[i2]:
                    i2+=1
                    res+=orig[i]
                else:
                    res = None
                    break
        else:
            break
    if res is not None:
        for i in range(i2, j2):
            res+=fix[i]
        return res
    orig = orig[::-1]
    fix1 = fix[::-1]
    i2 = 0
    j1 = len(orig)
    j2 = len(fix)
    res = ''
    for i in range(0, j1):
        if num1!=0 and i2<j2:
            if orig[i] not in good_symb:
                res+=orig[i]
                num1-=1
            else:
                if orig[i] == fix1[i2]:
                    i2+=1
                    res+=orig[i]
                else:
                    res = None
                    break
        else:
            break
    if res is not None:
        for i in range(i2, j2):
            res+=fix1[i]
        res = res[::-1]
        return res
    return fix

def preprocessing(tokens):
    res = []
    for token in tokens:
        if token == 'инстаграмм':
            res.append('инстаграм')
        elif token == 'филм':
            res.append('фильм')
        elif (token == 'youtu') or (token == 'yutub') or (token == 'ytebe') or (token == 'torube') or (token == 'yube') or (token == 'ytube'):
            res.append('youtube')
        elif (token == 'ютукьб') or (token == 'утюб') or (token == 'етюб') or (token == 'юдюп'):
            res.append('ютуб')
        elif token == 'кіного':
            res.append('киного')
        elif token == 'kinog':
            res.append('kinogo')
        elif token == 'шо':
            res.append('що')
        elif token == 'еротика':
            res.append('эротика')
        elif token == 'слизарио':
            res.append('слизерин')
        elif token == 'diepio':
            res.append('deepio')
        elif token == 'гоогл' or token == 'гуглл':
            res.append('гугл')
        elif token == 'мэил' or token == 'маил' or token == 'мейл':
            res.append('майл')
        elif token == 'slitherio':
            res.append('slytherin')
        else:
            res.append(token)
    return res