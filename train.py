import numpy as np
import string
from datetime import datetime
import random
import matplotlib.pyplot as plt
import os
import json


def remove_punct(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def encode_sentences():
    file = 'text.txt'
    sentences = []
    
    for line in open(file):
        s = remove_punct(line).lower().split()
        if len(s) > 1:
            sentences.append(s)
    
    indexed_sentences = []
    i = 2
    word2idx = {'START': 0, 'END': 1}
    idx2word = ['START', 'END']
    
    for sentence in sentences:
        indexed_sentence = []
        for token in sentence:
            if token not in word2idx:
                idx2word.append(token)
                word2idx[token] = i
                i = i + 1
            indexed_sentence.append(word2idx[token])
        indexed_sentences.append(indexed_sentence)
    
    return indexed_sentences, word2idx

def softmax(a):
    a = a - a.max()
    exp_a = np.exp(a)
    sigma = exp_a / exp_a.sum(axis = 1, keepdims = True)
    return sigma

def save_model(savedir, word2idx, W1, W2): 
    if not os.path.exists(savedir):
        os.mkdir(savedir)
        
    with open('%s/word2idx.json' % savedir, 'w') as f:
        json.dump(word2idx, f)
    
    np.savez('%s/weights.npz' % savedir, W1, W2)
                

if __name__ == "__main__":
    sentences, word2idx = encode_sentences()
    V = len(word2idx)
    print("Vocabulary Size: ", V)
    
    start_idx = word2idx['START']
    end_idx = word2idx['END']
        
    D = 100
    W1 = np.random.randn(V, D) / np.sqrt(V)
    W2 = np.random.randn(D, V) / np.sqrt(D)
    
    costs = []
    epochs = 5
    lr = 0.01
    
    for epoch in range(epochs):
        print("Epoch ", epoch)
        t0 = datetime.now()
        random.shuffle(sentences)
        
        j = 0
        for sentence in sentences:
            sentence = [start_idx] + sentence + [end_idx]
            n = len(sentence)
            
            inputs = sentence[:n - 1]
            targets = sentence[:1]
            
            hidden = np.tanh(W1[inputs])
            predictions = softmax(hidden.dot(W2))
            
            cost = -np.sum(np.log(predictions[np.arange(n - 1), targets])) / (n - 1)
            
            doutput = predictions
            doutput[np.arange(n - 1), targets] -= 1
            W2 = W2 - lr * hidden.T.dot(doutput)
            dhidden = doutput.dot(W2.T) * (1 - hidden*hidden)
            
            i = 0
            for w in inputs:
                W1[w] = W1[w] - lr * dhidden[i]
                i += 1
            
        print("Loss:", cost, ", Elapsed time:", datetime.now() - t0)
        costs.append(cost)
    plt.plot(costs)
    plt.title("Model Loss")
    
    save_model('chaucer_model', word2idx, W1, W2)