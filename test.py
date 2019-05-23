import numpy as np
import json
from sklearn.metrics.pairwise import pairwise_distances


def load_model(savedir):
    with open('%s/word2idx.json' % savedir) as f:
        word2idx = json.load(f)
    npz = np.load('%s/weights.npz' % savedir)
    W1 = npz['arr_0']
    W2 = npz['arr_1']
    return word2idx, W1, W2

def analogy(word, word2idx, idx2word, W):
    V, D = W.shape
    if word not in word2idx:
        print("Word '%s' not in vocabulary" % word)
    vector = W[word2idx[word]]
    output = []
    for i in range(100):
        distances = pairwise_distances(vector.reshape(1, D), W, metric = 'cosine').reshape(V)
        idx = distances.argsort()
        next_word = idx2word[idx[i + 1]]
        output.append(next_word)
        vector = W[word2idx[next_word]]
    print(' '.join(output))
    
    
def test_model(word2idx, W1, W2):
    W = (W1 + W2.T) / 2
    idx2word = {i:w for w, i in word2idx.items()}
    
    input_word = input("Enter a starting word: ")
    analogy(input_word, word2idx, idx2word, W)

if __name__ == "__main__":
    word2idx, W1, W2 = load_model('chaucer_model')
    test_model(word2idx, W1, W2)