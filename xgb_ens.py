import pandas as pd
import numpy as np
import xgboost as xgb


def xgb_acc_score(preds, dtrain):
    y_true = dtrain.get_label()
    y_pred = np.argmax(preds, axis=1)
    return [('acc', np.mean(y_true == y_pred))]


df_lr = pd.read_csv('./data/tfidf_stack_1w.csv')
df_dm = pd.read_csv('./data/dm_stack_1w.csv')
df_dbow = pd.read_csv('./data/dbow_stack_1w.csv')

df_lb = pd.read_csv('./data/all_v2.csv', usecols=['ID', 'Education', 'Age', 'Gender'], nrows=10000)
ys = {}
for lb in ['Education', 'Age', 'Gender']:
    ys[lb] = np.array(df_lb[lb])

df = pd.concat([df_lr, df_dm, df_dbow], axis=1)
print(df.columns)

TR = 8000
df_sub = pd.DataFrame()
df_sub['ID'] = df_lb.iloc[TR:]['ID']
seed = 10

# -------------------------education----------------------------------
lb = 'Education'
print(lb)

esr = 25
evals = 1
n_trees = 1000

num_class = len(pd.value_counts(ys[lb]))
X = df.iloc[:TR]
y = ys[lb][:TR]
X_te = df.iloc[TR:]
y_te = ys[lb][TR:]

ss = 0.9
mc = 2
md = 8
gm = 2

params = {
    'objective': 'multi:softprob',
    'booster': 'gbtree',
    'num_class': num_class,
    'max_depth': md,
    'min_child_weight': mc,
    'subsample': ss,
    'colsample_bytree': 0.8,
    'gamma': gm,
    'eta': 0.01,
    'lambda': 0,
    'alpha': 0,
    'silent': 1
}

dtrain = xgb.DMatrix(X, y)
dvalid = xgb.DMatrix(X_te, y_te)
watchlist = [(dtrain, 'dtrain'), (dvalid, 'eval')]
bst = xgb.train(params, dtrain, n_trees, evals=watchlist, feval=xgb_acc_score, maximize=True,
                early_stopping_rounds=esr, verbose_eval=evals)
df_sub['Education'] = np.argmax(bst.predict(dvalid), axis=1) + 1

# ------------------------ age-----------------------------------
lb = 'Age'
print(lb)
num_class = len(pd.value_counts(ys[lb]))

num_class = len(pd.value_counts(ys[lb]))
X = df.iloc[:TR]
y = ys[lb][:TR]
X_te = df.iloc[TR:]
y_te = ys[lb][TR:]

ss = 0.5
mc = 3
md = 7
gm = 2

params = {
    "objective": "multi:softprob",
    "booster": "gbtree",
    "num_class": num_class,
    'max_depth': md,
    'min_child_weight': mc,
    'subsample': ss,
    'colsample_bytree': 1,
    'gamma': gm,
    "eta": 0.01,
    "lambda": 0,
    'alpha': 0,
    "silent": 1,
}

dtrain = xgb.DMatrix(X, y)
dvalid = xgb.DMatrix(X_te, y_te)
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
bst = xgb.train(params, dtrain, n_trees, evals=watchlist, feval=xgb_acc_score, maximize=True,
                early_stopping_rounds=esr, verbose_eval=evals)
df_sub['Age'] = np.argmax(bst.predict(dvalid), axis=1)+1

# --------------------------gender-------------------------------------
lb = 'Gender'
print(lb)
num_class = len(pd.value_counts(ys[lb]))

num_class = len(pd.value_counts(ys[lb]))
X = df.iloc[:TR]
y = ys[lb][:TR]
X_te = df.iloc[TR:]
y_te = ys[lb][TR:]

ss = 0.5
mc = 0.8
md = 7
gm = 1

params = {
    "objective": "multi:softprob",
    "booster": "gbtree",
    "num_class": num_class,
    'max_depth': md,
    'min_child_weight': mc,
    'subsample': ss,
    'colsample_bytree': 1,
    'gamma': gm,
    "eta": 0.01,
    "lambda": 0,
    'alpha': 0,
    "silent": 1,
}

dtrain = xgb.DMatrix(X, y)
dvalid = xgb.DMatrix(X_te, y_te)
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
bst = xgb.train(params, dtrain, n_trees, evals=watchlist, feval=xgb_acc_score, maximize=True,
                early_stopping_rounds=esr, verbose_eval=evals)
df_sub['Gender'] = np.argmax(bst.predict(dvalid), axis=1) + 1

# df_sub = df_sub[['ID','Age','Gender','Education']]  # 不知道为什么要加这一句，这样就将三个属性直接打乱了
df_sub.to_csv('./data/tfidf_dm_dbow_2k.csv', index=None)

