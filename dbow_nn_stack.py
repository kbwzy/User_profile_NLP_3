import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from gensim.models import Doc2Vec

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.utils import to_categorical


def myAcc(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    return np.mean(y_true == y_pred)


df_all = pd.read_csv('./data/all_v2.csv', usecols=['ID', 'Education', 'Age', 'Gender'], nrows=10000)
ys = {}
for label in ['Education', 'Age', 'Gender']:
    ys[label] = np.array(df_all[label])

model = Doc2Vec.load('./data/dbow_model.model')
X_sp = np.array([model.docvecs[i] for i in range(len(df_all))])

df_stack = pd.DataFrame(index=range(len(df_all)))
TR = 8000
n = 5
X = X_sp[:TR]
X_te = X_sp[TR:]

for i, lb in enumerate(['Education', 'Age', 'Gender']):
    print(lb)
    num_class = len(pd.value_counts(ys[lb]))
    y = ys[lb][:TR]
    y_te = ys[lb][TR:]

    stack = np.zeros((X.shape[0], num_class))
    stack_te = np.zeros((X_te.shape[0], num_class))

    nn_model = Sequential()
    nn_model.add(Dense(300, input_shape=(X.shape[1],)))
    nn_model.add(Dropout(0.1))
    nn_model.add(Activation('tanh'))
    nn_model.add(Dense(num_class))
    nn_model.add(Activation('softmax'))

    nn_model.compile(
        optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

    kfold = KFold(n_splits=n, random_state=2019)
    for i, (tr, va) in enumerate(kfold.split(X)):
        print('stack:{}/{}'.format(i + 1, n))
        nb_classes = num_class
        X_train = X[tr].astype('float32')
        y_train = y[tr]
        Y_train = to_categorical(y_train, num_classes=nb_classes)
        X_test = X_te.astype('float32')
        y_test = y_te
        Y_test = to_categorical(y_test, num_classes=nb_classes)

        history = nn_model.fit(X_train, Y_train,
                               shuffle=True, batch_size=128, epochs=16, verbose=2,
                               validation_data=(X_test, Y_test))

        y_pred_va = nn_model.predict_proba(X[va])
        y_pred_te = nn_model.predict_proba(X_te)
        print('va acc: ', myAcc(y[va], y_pred_va))
        print('te acc: ', myAcc(y_te, y_pred_te))
        stack[va] += y_pred_va
        stack_te += y_pred_te
    stack_te /= n
    stack_all = np.vstack([stack, stack_te])
    for i in range(stack_all.shape[1]):
        df_stack['dbow_nn_{}_{}'.format(lb, i)] = stack_all[:, i]

df_stack.to_csv('./data/dbow_stack_1w.csv', index=None, encoding='utf-8')
print('done!')

