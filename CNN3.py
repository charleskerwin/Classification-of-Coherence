import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout, BatchNormalization, MaxPool1D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix


d = 5
ks = [2, 3, 4, 5]
data = [np.loadtxt(f'data/{d}_{k}.csv', dtype=complex) for k in ks]
data = np.array(data)

if data.shape != (4, 1.25e6, 50):
  raise Exception(f'Shape {data.shape} is not expected.')


n_examples = data.shape[1]
n_features = data.shape[2]
n_classes = len(ks)


# one hot encoding
Y = np.zeros((n_classes * n_examples, n_classes))

for k_index in range(len(ks)):
    s_i = k_index * n_examples # start index
    s_f = (k_index + 1) * n_examples # end index
    Y[s_i:s_f, k_index] += 1 # matrix of examples x classes


# matrix of examples x features
X = data.reshape(n_classes * n_examples, n_features)


# shuffle the dataset

X_df = pd.DataFrame(X)
Y_df = pd.DataFrame(Y)
XY = pd.concat([X_df, Y_df], axis=1)
XY = XY.sample(frac=1)
X = XY.iloc[:,:n_features]
Y = XY.iloc[:,n_features:]


# test set is 20% of the data

train_size = int(.8 * X.shape[0])

X_test = X.iloc[train_size:]
Y_test = Y.iloc[train_size:]

X_train = X.iloc[:train_size]
Y_train = Y.iloc[:train_size]

X_train_final = X_train.copy()
Y_train_final = Y_train.copy()


# validation set is 20% of 80% = 16% of the data

validation_size = int(.8 * X_train.shape[0])

X_val = X_train[validation_size:]
Y_val = Y_train[validation_size:]

X_train = X_train.iloc[:validation_size]
Y_train = Y_train.iloc[:validation_size]


# convert dataframes to numpy arrays

X_train = X_train.to_numpy()
Y_train = Y_train.to_numpy()

X_val = X_val.to_numpy()
Y_val = Y_val.to_numpy()

X_train_final = X_train_final.to_numpy()
Y_train_final = Y_train_final.to_numpy()

X_test = X_test.to_numpy()
Y_test = Y_test.to_numpy()


# reshape X data for CNN

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_train_final = X_train_final.reshape(X_train_final.shape[0], X_train_final.shape[1], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

lr = 1e-3
adam = Adam(learning_rate=lr)

epochs = 200
batch_size = 1024

do_rate = .2

with tf.device('/device:GPU:0'):

  model = Sequential()

  model.add(Conv1D(filters=64, kernel_size=25, activation="elu", strides=1,
                   padding="same", input_shape=(n_features, 1)))

  model.add(MaxPool1D(pool_size=2))

  model.add(Conv1D(filters=128, kernel_size=3, activation="elu", strides=1,
                    padding="same"))

  model.add(Conv1D(filters=128, kernel_size=3, activation="elu", strides=1,
                    padding="same"))

  model.add(MaxPool1D(pool_size=2))

  model.add(Conv1D(filters=256, kernel_size=3, activation="elu", strides=1,
                    padding="same"))

  model.add(Conv1D(filters=256, kernel_size=3, activation="elu", strides=1,
                    padding="same"))

  model.add(MaxPool1D(pool_size=2))

  model.add(Flatten())

  model.add(Dense(128, activation="elu"))

  model.add(BatchNormalization())

  model.add(Dropout(do_rate))

  model.add(Dense(64, activation="elu"))

  model.add(BatchNormalization())

  model.add(Dropout(do_rate))

  model.add(Dense(32, activation="elu"))

  model.add(BatchNormalization())

  model.add(Dropout(do_rate))

  model.add(Dense(4, activation="softmax"))


  history = model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
  model.summary()

  history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
                                  validation_data=(X_val, Y_val), verbose=1)


# learning curves

def plot_lc(name):

    sns.set_style(style='darkgrid')

    plt.figure()
    pd.DataFrame(history.history).plot()
    plt.xlabel(f'Epoch (batch size={batch_size})')
    plt.savefig(f'results/learning_curves/{name}')
    plt.close()


plot_lc('tr_val')


# predict

Y_pred = model.predict(X_train, verbose=1)


# confusion matrix

Y_train = np.array(Y_train)
y_train_vec = np.zeros(len(Y_train))
y_pred_vec = np.zeros(len(Y_pred))

for i in range(len(Y_pred)):

    max_index = np.argmax(Y_train[i])
    y_train_vec[i] = max_index + ks[0]

    max_index = np.argmax(Y_pred[i])
    y_pred_vec[i] = max_index + ks[0]

def plot_cms(name, y, y_pred):

    plt.figure()
    cm = pd.crosstab(y, y_pred, rownames=['True Coherence Rank'], colnames=['Predicted Coherence Rank'])
    sns.heatmap(cm, annot=True).figure.savefig(f'results/confusion_matrices/{name}')
    plt.close()


    plt.figure()
    cm_norm = pd.crosstab(y, y_pred, rownames=['True Coherence Rank'], colnames=['Predicted Coherence Rank'], normalize=True)
    sns.heatmap(cm_norm, annot=True).figure.savefig(f'results/confusion_matrices/{name}_norm')
    plt.close()

plot_cms('tr_val', y_train_vec, y_pred_vec)


# train on both training and validation sets

with tf.device('/device:GPU:0'):
    history = model.fit(X_train_final, Y_train_final, epochs=50, batch_size=batch_size, verbose=1)


# learning curves

plot_lc('final_tr')


# predict

Y_pred = model.predict(X_train_final, verbose=1)


# confusion matrix

Y_train_final = np.array(Y_train_final)
y_train_vec = np.zeros(len(Y_train_final))
y_pred_vec = np.zeros(len(Y_pred))

for i in range(len(Y_pred)):

    max_index = np.argmax(Y_train_final[i])
    y_train_vec[i] = max_index + ks[0]

    max_index = np.argmax(Y_pred[i])
    y_pred_vec[i] = max_index + ks[0]


# plot confusion matrices

plot_cms('final_tr', y_train_vec, y_pred_vec)


# evaluate

evaluation = model.evaluate(X_test, Y_test)

with open('results/accuracy.txt','w') as file:
    file.write('Loss: {}\tAccuracy: {}'.format(evaluation[0], evaluation[1]))


# save model

model.save('results/model.h5')
