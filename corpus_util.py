import pandas as pd
import sys

path = "corpus/"
file_name = "weibo_senti_100k.csv"

try: 
    pd_all = pd.read_csv(path + file_name)
except:
    print(FileNotFoundError("File '" + file_name + "'" " is not found in " + path))
    sys.exit(-1)

pd_positive = pd_all[pd_all.label == 1]
pd_negative = pd_all[pd_all.label == 0]

def get_balanced_corpus(expected_size, pos_corpus = pd_positive, neg_corpus = pd_negative):
    size = expected_size // 2
    balanced_pos_corpus = pos_corpus.sample(size, replace = pos_corpus.shape[0] < size)
    balanced_neg_corpus = neg_corpus.sample(size, replace = neg_corpus.shape[0] < size)
    return balanced_pos_corpus, balanced_neg_corpus