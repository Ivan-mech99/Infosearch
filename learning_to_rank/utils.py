import numpy as np

def build_q_to_doc(q_ids):
    num_doc = -1
    q_to_doc = dict()
    for elem in q_ids:
        if elem not in q_to_doc:
            q_to_doc[elem] = []
            num_doc += 1
            q_to_doc[elem].append(num_doc)
        else:
            num_doc += 1
            q_to_doc[elem].append(num_doc)
    return q_to_doc

def calc_idcg(reli, num_top):
    if np.isclose(np.sum(reli), 0.0):
        return 1.0
    reli_sort = np.sort(reli)
    reli_sort = reli_sort[::-1]
    if reli_sort.shape[0] < num_top:
        dim1 = reli_sort.shape[0]
    else:
        dim1 = num_top
    reli_sort = reli_sort[:num_top]
    denominator = np.array([1.0/np.log2(i+1) for i in range(1, len(reli_sort)+1)])
    numerator = np.array([2**rel-1 for rel in reli_sort])
    return np.sum(numerator*denominator)

def build_S_mat(q_to_doc, y):
    S_list = dict()
    for key in q_to_doc:
        marks = y[np.array(q_to_doc[key])]
        dim = len(marks)
        S_ij = np.zeros((dim, dim))
        S_ij += ((marks.reshape(-1, 1) - marks) > 0).astype(int)
        S_ij -= ((marks.reshape(-1, 1) - marks) < 0).astype(int)
        S_list[key] = S_ij
    return S_list

def build_idcg_dict(q_to_doc, y, num_top):
    query_idcg = dict()
    for q_id in q_to_doc:
        idcg = calc_idcg(y[q_to_doc[q_id]], num_top)
        query_idcg[q_id] = idcg
    return query_idcg
            
def update_leafs(X, tree, gradient, hessian, lambda_ = 0):
    leafs_data = dict()
    cur_obj = -1
    for leaf in tree.tree_.apply(X):
        if leaf not in leafs_data:
            leafs_data[leaf] = []
            cur_obj+=1
            leafs_data[leaf].append(cur_obj)
        else:
            cur_obj+=1
            leafs_data[leaf].append(cur_obj)
    for leaf in leafs_data:
        numerator = - np.sum(gradient[leafs_data[leaf]])
        denominator = np.sum(hessian[leafs_data[leaf]])
        if np.isclose(denominator, 0):
            tree.tree_.value[leaf] = 0.0
        else:
            tree.tree_.value[leaf] = numerator / (denominator + lambda_)
    return tree

def calc_delta_ndcg(cur_true, cur_pred, val):
    ordered = np.zeros(len(cur_pred))
    indicies = np.argsort(-cur_pred)
    ordered[indicies] = np.arange(1, len(cur_pred) + 1)
    delta_ndcg = (((2**cur_true.reshape(-1, 1) - 2**cur_true) * (1 / np.log2(1 + ordered.reshape(-1, 1)) - 1 / np.log2(1 + ordered))) / val)
    return np.abs(delta_ndcg)

def calc_sigmoid(cur_pred, s_cur):
    delta_preds = cur_pred.reshape(-1, 1) - cur_pred
    sigmoid = np.zeros(delta_preds.shape)
    sigmoid += 1.0/(1.0+np.exp(delta_preds)) * (s_cur > 0)
    sigmoid += 1.0/(1.0+np.exp(-delta_preds)) * (s_cur < 0)
    return sigmoid