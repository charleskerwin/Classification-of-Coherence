import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorflow.keras.models import load_model

model = load_model('model.h5')

d = 5
ks = [2, 3, 4, 5]
data = [np.loadtxt(f'data/{d}_{k}.csv', dtype=complex) for k in ks]
data = np.array(data)

if data.shape != (4, 11, 50):
  raise Exception(f'Shape {data.shape} is not expected.')

n_examples = data.shape[1]
n_features = data.shape[2]
n_classes = len(ks)


X = data.reshape(n_classes * n_examples, n_features, 1)

# one hot encoding
Y = np.zeros((n_classes * n_examples, n_classes))

for k_index in range(len(ks)):
    s_i = k_index * n_examples # start index
    s_f = (k_index + 1) * n_examples # end index
    Y[s_i:s_f, k_index] += 1 # matrix of examples x classes


Y_pred = model.predict(X, verbose=1)

y = np.argmax(Y, axis=1).reshape(-1, 1) + ks[0]
y_pred = np.argmax(Y_pred, axis=1).reshape(-1, 1) + ks[0]


correct_pred = np.array(y == y_pred)

np.savetxt('results/correct_predictions.csv', np.concatenate((y, y_pred, correct_pred), axis=1))

correct_pred = correct_pred.reshape(n_classes, n_examples)
correct_pred = np.flip(correct_pred, axis=0)

plt.figure()

xticks = np.around(np.array([0, 0.5, 0.9, 0.95, 0.99, 0.995, 0.996, 0.997, 0.998, 0.999, 1]), 3)
yticks = ks[::-1]

ax = sns.heatmap(correct_pred, annot=False, cbar=False, square=True, xticklabels=xticks, yticklabels=yticks, linewidths=1)

bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)

ax.set_xlabel('$p$')
ax.set_ylabel('True Coherence Rank, $k$')

plt.savefig(f'results/plot')


fig, ax = plt.subplots()

ax.set_ylim(1.5, 7)

xs = np.arange(0, 1.1, .1)

ax.set_xticks(xs)
ax.set_xticklabels(xticks)

y_pred = y_pred.flatten()

colours = ('yellow', 'red', 'blue', 'orange')

for k in ks:
    ax.axhline(k, color='green', lw=5)

for i, k in enumerate(ks):

    col = colours[i]

    s_i = i * 11
    s_f = (i + 1) * 11

    ys = y_pred[s_i:s_f]

    plt.plot(xs, ys, color=col, label=f'$k = {k}$')

ax.set_xlabel('$p$')
ax.set_ylabel('Predicted Coherence Rank')

plt.legend()
plt.savefig(f'results/lineplot')
