from corpus_util import *
from stopword_util import *
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

EXPECTED_SIZE = 10000
N_COMPONENTS = 100

def main():
    pos_corpus, neg_corpus = get_balanced_corpus(EXPECTED_SIZE)
    stopwords = getStopwords()

    pos_txt = preprocessCorpus(pos_corpus, stopwords)
    neg_txt = preprocessCorpus(neg_corpus, stopwords)

    posInput = getTxtVec(pos_txt, get_model(pos_txt))
    negInput = getTxtVec(neg_txt, get_model(neg_txt))

    y = np.concatenate((np.ones(len(posInput)), np.zeros(len(negInput))))

    X = posInput[:]
    for neg in negInput:
        X.append(neg)
    X = np.array(X)

    ### Plotting test
    pca = PCA(n_components=N_COMPONENTS)
    pca.fit(X)
    plt.figure(1, figsize=(4, 3))
    plt.clf()
    plt.axes([.2, .2, .7, .7])
    plt.plot(pca.explained_variance_, linewidth=2)
    plt.axis('tight')
    plt.xlabel('n_components')
    plt.ylabel('explained_variance_')
    plt.show()
    ###
    
def getWordVec(words, model):
    vecs = []
    for word in words:
        word = word.replace('\n', '')
        try:
            vecs.append(model[word])
        except:
            continue
    return np.array(vecs, dtype='float')

def getTxtVec(txt, model):
    vecs = []
    for line in txt:
        words = line.split(' ')
        word_vecs = getWordVec(words, model)
        if len(word_vecs) > 0:
            vecArr = sum(np.array(word_vecs)) / len(word_vecs)
            vecs.append(vecArr)
    return vecs

def get_model(corpus):
    model = Word2Vec(corpus)
    return model

if __name__ == '__main__':
    main()