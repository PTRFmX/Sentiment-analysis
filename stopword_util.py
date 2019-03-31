import jieba
import sys
import codecs

def getStopwords():
    stopword_file = "stop_words_source/chinese_stop_words.txt"
    stopwords = set(())
    for word in codecs.open(stopword_file, 'r').readlines():
        stopwords.add(word.strip())
    return stopwords

def preprocessSen(line, stopwords):
    words = jieba.cut(line, cut_all=False)
    sen = ""
    for word in words:
        word = word.strip()
        if word not in stopwords:
            sen += word + " "
    return sen

def preprocessCorpus(corpus, stopwords):
    txt = corpus.review
    processed_txt = ""
    for line in txt:
        processed_line = preprocessSen(line, stopwords)
        processed_txt += processed_line + '\n'
    return processed_txt