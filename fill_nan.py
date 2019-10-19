# 作者将三个属性中为0（代表未知）的数据进行处理，通过不为0的数据使用logistic regression进行预测后填充
# query的特征表示使用TF-IDF
# 处理完的文件为all_v2.csv
import pandas as pd
import numpy as np
import jieba
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# 读取原始数据，只取前1w条
df_tr = pd.read_csv('./data/train.csv', sep="###__###", header=None, nrows=10000)
df_tr.columns = ['ID', 'Age', 'Gender', 'Education', 'query']
df_te = pd.read_csv('./data/test.csv', sep="###__###", header=None, nrows=10000)
df_te.columns = ['ID', 'query']
print(df_tr.shape)
print(df_te.shape)

df_all = df_tr
for lb in ['Education', 'Age', 'Gender']:
    df_all[lb] = df_all[lb] - 1
    print(df_all.iloc[:10000][lb].value_counts())


class Tokenizer:
    def __init__(self):
        self.n = 0

    def __call__(self, line, *args, **kwargs):
        tokens = []
        for query in line.split('\t'):
            words = [word for word in jieba.cut(query)]
            for gram in [1, 2]:
                for i in range(len(words) - gram + 1):
                    tokens += ['_*_'.join(words[i:i+gram])]
        if np.random.rand() < 0.00001:
            print(line)
            print('='*20)
            print(tokens)
        self.n += 1
        if self.n % 1000 == 0:
            print(self.n, end=' ')
        return tokens


tfv = TfidfVectorizer(tokenizer=Tokenizer(), min_df=3, max_df=0.95, sublinear_tf=True)
X_sp = tfv.fit_transform(df_all['query'])
print(len(tfv.vocabulary_))
X_all = X_sp

lb = 'Education'
idx = 3
tr = np.where(df_all[lb] != -1)[0]
va = np.where(df_all[lb] == -1)[0]
df_all.iloc[va, idx] = \
    LogisticRegression(C=1).fit(X_all[tr], df_all.iloc[tr, idx]).predict(X_all[va])

lb = 'Age'
idx = 1
tr = np.where(df_all[lb] != -1)[0]
va = np.where(df_all[lb] == -1)[0]
df_all.iloc[va, idx] = \
    LogisticRegression(C=2).fit(X_all[tr], df_all.iloc[tr, idx]).predict(X_all[va])

lb = 'Gender'
idx = 2
tr = np.where(df_all[lb]!=-1)[0]
va = np.where(df_all[lb]==-1)[0]
df_all.iloc[va,idx] = \
    LogisticRegression(C=2).fit(X_all[tr], df_all.iloc[tr, idx]).predict(X_all[va])

df_all = pd.concat([df_all, df_te]).fillna(0)
df_all.to_csv('./data/all_v2.csv', index=None, encoding='utf8')

