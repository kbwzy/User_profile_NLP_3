import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


# -------------定义评估函数---------------
def myAcc(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    return np.mean(y_true == y_pred)


# load data
df_all = pd.read_csv('./data/all_v2.csv', encoding='utf-8', nrows=10000)
ys = {}
for lb in ['Education', 'Age', 'Gender']:
    ys[lb] = np.array(df_all[lb])

# ********************************************************************************
# ================================================ #
# 对query使用TF-IDF进行表示，保存为tfidf_1w.feat文件.  #
# 该过程较为耗时，无需反复执行，运行一次后即可注释掉      #
# 再次使用直接用pickle读取                            #
# ================================================ #
# class Tokenizer:
#     def __init__(self):
#         self.n = 0
#
#     def __call__(self, line, *args, **kwargs):
#         tokens = []
#         for query in line.split('\t'):
#             words = [word for word in jieba.cut(query)]
#             for gram in [1, 2]:
#                 for i in range(len(words) - gram + 1):
#                     tokens += ['_*_'.join(words[i:i+gram])]
#         self.n += 1
#         if self.n % 1000 == 0:
#             print(self.n)
#         return tokens
#
#
# tfv = TfidfVectorizer(tokenizer=Tokenizer(), min_df=3, max_df=0.95, sublinear_tf=True)
# X_sp = tfv.fit_transform(df_all['query'])
# pickle.dump(X_sp, open('./data/tfidf_1w.pkl', 'wb'))
# ********************************************************************************

X_sp = pickle.load(open('./data/tfidf_1w.feat', 'rb'))
df_stack = pd.DataFrame(index=range(len(df_all)))

# -----------------------stack for education/age/gender------------------
for lb in ['Education', 'Age', 'Gender']:
    print(lb)
    TR = 8000
    num_class = len(pd.value_counts(ys[lb]))
    n = 5

    X = X_sp[:TR]
    y = ys[lb][:TR]
    X_te = X_sp[TR:]
    y_te = ys[lb][TR:]

    stack = np.zeros((X.shape[0], num_class))
    stack_te = np.zeros((X_te.shape[0], num_class))

    kfold = KFold(n_splits=n, random_state=2019)
    for i, (tr, va) in enumerate(kfold.split(X)):
        print('stack:%d/%d' % (i + 1, n))
        clf = LogisticRegression(C=3)
        clf.fit(X[tr], y[tr])
        y_pred_va = clf.predict_proba(X[va])
        y_pred_te = clf.predict_proba(X_te)
        print('va acc:', myAcc(y[va], y_pred_va))
        print('te acc:', myAcc(y_te, y_pred_te))
        stack[va] += y_pred_va
        stack_te += y_pred_te
    stack_te /= n
    stack_all = np.vstack([stack, stack_te])
    for i in range(stack_all.shape[1]):
        df_stack['tfidf_{}_{}'.format(lb, i)] = stack_all[:, i]

df_stack.to_csv('./data/tfidf_stack_1w.csv', index=None, encoding='utf-8')
print('done!')