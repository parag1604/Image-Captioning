import numpy as np
import matplotlib.pyplot as plt


def ma(scores, window=5):
    ma = np.zeros(len(scores))
    for i in range(len(scores)):
        if i < window:
            ma[i] = np.mean(scores[:i+1])
        else:
            ma[i] = np.mean(scores[i-window+1:i+1])
    return ma

scores1 = np.load('temp/vit_lstm_val_scores.npy')
scores2 = np.load('temp/vit_lstm_attn_val_scores.npy')
scores3 = np.load('temp/vit_lstm_attnb_val_scores.npy')
scores4 = np.load('temp/vit_tf_val_scores.npy')

plt.plot(ma(scores1), label='Vanilla LSTM')
plt.plot(ma(scores2), label='Attention')
plt.plot(ma(scores3), label='Bahdanau Attention')
plt.plot(ma(scores4), label='Transformer')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('BLEU Score')
plt.savefig('images/plot.png', bbox_inches='tight')
plt.close()
