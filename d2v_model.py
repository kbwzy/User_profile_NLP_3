# 训练doc2vec模型，dm和dbow都进行了训练
import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# ********************************************************************************
# =========================================== #
# 对query进行分词, 保存为query_cut.csv文件。     #
# 该过程较为耗时，无需反复执行，运行一次后即可注释掉 #
# 再次使用直接读取                              #
# =========================================== #
# df_query = pd.read_csv('./data/all_v2.csv', usecols=['query'], encoding='utf-8')
# rows = []
# for i, line in enumerate(df_query.iloc[:10000]['query']):
#     words = []
#     row = {}
#     for query in line.split('\t'):
#         words.extend(list(jieba.cut(query)))
#     row['tag'] = i
#     row['query_cut'] = words
#     rows.append(row)
# df_query_cut = pd.DataFrame(rows)
# df_query_cut.to_csv('./data/query_cut.csv', index=None, encoding='utf8')
# ***********************************************************************************

df_query_cut = pd.read_csv('./data/query_cut.csv', encoding='utf-8')
query_tagged = df_query_cut.apply(
    lambda line: TaggedDocument(words=line['query_cut'], tags=[line['tag']]), axis=1)

df_lb = pd.read_csv('./data/all_v2.csv',
                    usecols=['Education', 'Age', 'Gender'], nrows=10000)
ys = {}
for lb in ['Education', 'Age', 'Gender']:
    ys[lb] = np.array(df_lb[lb])

# -------------------train dbow doc2vec---------------------------------------------
dbow_model = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=3,
                     window=30, sample=1e-5, workers=8, alpha=0.025, min_alpha=0.025)
dbow_model.build_vocab(query_tagged)

for i in range(2):
    print('pass:', i + 1)
    dbow_model.train(query_tagged, total_examples=dbow_model.corpus_count, epochs=dbow_model.epochs)
    X_d2v = np.array([dbow_model.docvecs[i] for i in range(10000)])
    for lb in ['Education', 'Age', 'Gender']:
        scores = cross_val_score(LogisticRegression(C=3), X_d2v, ys[lb], cv=5)
        print('dbow', lb, scores, np.mean(scores))
dbow_model.save('./data/dbow_model.model')

print('='*20)

# -------------------train dm doc2vec---------------------------------------------
dm_model = Doc2Vec(dm=1, vector_size=300, negative=5, hs=0, min_count=3,
                   window=10, sample=1e-5, workers=8, alpha=0.025, min_alpha=0.025)
dm_model.build_vocab(query_tagged)

for i in range(2):
    print('pass:', i + 1)
    dm_model.train(query_tagged, total_examples=dm_model.corpus_count, epochs=dm_model.epochs)
    X_d2v = np.array([dm_model.docvecs[i] for i in range(10000)])
    for lb in ['Education', 'Age', 'Gender']:
        scores = cross_val_score(LogisticRegression(C=3), X_d2v, ys[lb], cv=5)
        print('dm', lb, scores, np.mean(scores))
dm_model.save('./data/dm_model.model')