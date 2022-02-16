import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_svmlight_file
from utils import *

class LambdaRanker:

    def __init__(self, n_estimators, max_depth, learning_rate, num_top, lambda_fine = 0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_top = num_top
        self.lambda_fine = lambda_fine
        self.ensemble = []

    def load_train_data(self):
        X, y, q_ids = load_svmlight_file('train.txt', query_id=True)
        self.nonzero_cols = np.where(np.sum(X, axis = 0) != 0)[1]
        self.X_train = X[:, self.nonzero_cols].astype(np.float32)
        self.y_train = y
        self.query_doc = build_q_to_doc(q_ids)
        self.query_idcg = build_idcg_dict(self.query_doc, self.y_train, self.num_top)
        self.S_ij = build_S_mat(self.query_doc, self.y_train)
        
    def load_test_data(self):
        X, y, q_ids = load_svmlight_file('test.txt', query_id=True)
        self.X_test = X[:, self.nonzero_cols].astype(np.float32)
        self.q_ids_test = q_ids
        self.test_query_doc = build_q_to_doc(self.q_ids_test)

    def calc_grad(self, true_res, pred_res):
        hessian = np.zeros(len(true_res))
        gradient = np.zeros(len(true_res))
        for key in self.query_doc:
            docs = self.query_doc[key]
            cur_true = true_res[docs]
            cur_pred = pred_res[docs]
            delta_ndcg = calc_delta_ndcg(cur_true, cur_pred, self.query_idcg[key])
            s_cur = self.S_ij[key]
            sigmoid = calc_sigmoid(cur_pred, s_cur)
            lambda_ij = - delta_ndcg * sigmoid
            loc_grad = np.sum(lambda_ij * s_cur, axis=1)
            gradient[docs] = loc_grad
            hessian[docs] = np.sum(delta_ndcg * sigmoid * (1 - sigmoid) * (s_cur != 0), axis=1)
        return gradient, hessian

    def fit(self):
        predictions = np.zeros(len(self.y_train))
        for i in tqdm(range(self.n_estimators)):
            tree = DecisionTreeRegressor(max_depth=self.max_depth, max_features = 'sqrt')
            gradient, hessian = self.calc_grad(self.y_train, predictions)
            tree.fit(self.X_train, -gradient)
            tree = update_leafs(self.X_train, tree, gradient, hessian, self.lambda_fine)
            self.ensemble.append(tree)
            predictions += self.learning_rate * tree.predict(self.X_train)

    def predict(self):
        shape1 = self.ensemble[0].predict(self.X_test)
        res = np.zeros(len(shape1))
        for tree in tqdm(self.ensemble):
            res += self.learning_rate * tree.predict(self.X_test)
        return res
    
    def predict_partly(self, lim=5):
        shape1 = self.ensemble[0].predict(self.X_test)
        res = np.zeros(len(shape1))
        for i in tqdm(range(0, lim)):
            res += self.learning_rate * self.ensemble[i].predict(self.X_test)
        return res
    
    def output_res(self, fname, preds):
        submit = pd.DataFrame()
        submit['QueryId'] = self.q_ids_test
        submit['DocumentId'] = np.arange(1, submit.shape[0] + 1)
        submit['pred'] = preds
        keys = submit[['QueryId']].drop_duplicates()
        values = submit.sort_values(by=['QueryId', 'pred'], ascending=False)
        ans = keys.merge(values)
        ans = ans[['QueryId', 'DocumentId']]
        ans.to_csv(fname, index=False) 
    
