import requests
import json

from gensim.models.word2vec import Word2Vec
import gensim
import logging
import nltk
import os
import time
import pickle
from nltk.stem import WordNetLemmatizer
from konlpy.tag import Twitter
from nltk.tag import pos_tag

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def extractData(url):
    page = requests.get(url).json()
    if ('content' in page['_source']):
        content = page['_source']['content']
        # print(content)
        if ('AAAAAAAAAAAA' in content) or ('Error occurred' in content):
            return ""
        return content
    else:
        return ""

def extractID(url):
    page = requests.get(url).json()
    f = open('test_data_180000.txt', mode='w', encoding='utf8')

    for index, item in enumerate(page["hits"]["hits"]):
        data = extractData('http://192.168.101.183:9200/wrapsody/document/' + item['_id'])

        if (index % 100 == 0):
            logging.info("extract {0} id".format(index))
        # print(item['_id'])
        f.write("%s\n" % data)

def tokenize(dict):
    list = []
    lm = WordNetLemmatizer()
    for tp in dict:
        if (tp[1] == 'Noun'):
            list.append(tp[0])

        if (tp[1] == 'Alpha'):
            tag = pos_tag([tp[0]])
            if "NN" in tag[0][1]:
                list.append(lm.lemmatize(tp[0]).lower())
                continue
            if "VB" in tag[0][1]:
                list.append(lm.lemmatize(tp[0], "v").lower())
                continue
            if "JJ" in tag[0][1]:
                list.append(lm.lemmatize(tp[0], "a").lower())
                continue
            if "RB" in tag[0][1]:
                list.append(lm.lemmatize(tp[0], "r").lower())
                continue

    return list

def read_input(input_file):
    f = open(input_file, 'r', encoding='utf-8', errors='ignore')
    content= f.readlines()
    # twitter = Twitter()

    for i, line in enumerate(content):
        # dict = twitter.pos(line.encode('utf8').decode('utf8'))
        # print(line)
        # tk = tokenize(dict)
        # tk = line.lower().split(" ")

        if (i % 100 == 0):
            logging.info("read {0} reviews".format(i))

        # do some pre-processing and return a list of words for each review text
        yield gensim.utils.simple_preprocess(line)

def train(document):
    f = open('save_model_04.txt', mode='w', encoding='utf8')
    if os.stat('save_model_04.txt').st_size == 0:
        logging.info("If this is seen, that means you got the right approach!")
        update = False
        model = gensim.models.Word2Vec(sg=1, size=150, window=5, workers=3, batch_words=10000, min_count=5)
    else:
        update = True
        model = gensim.models.Word2Vec.load('save_model_04.txt')

    model.build_vocab(document, keep_raw_vocab=False, trim_rule=None, progress_per=10000, update=update)
    model.train(document, total_examples=len(document), epochs=15)
    model.save('save_model_04.txt')

def test():
    model = gensim.models.Word2Vec.load('save_model_04.txt')
    w1 = "파수"
    print(model.wv.most_similar(positive=w1, topn=6))

def main():
    # url = 'http://192.168.101.183:9200/wrapsody/document/_search?size=180000&_source=false&sort=_id:asc'
    # start_time = time.time()
    # extractID(url)
    # logging.info("Done extracting data!")
    # print("--- %s seconds ---" % (time.time() - start_time))

    # start_time = time.time()
    # document = list(read_input('test_data_180000.txt'))
    # print("--- %s seconds ---" % (time.time() - start_time))
    # logging.info("Done reading data file!")

    # start_time = time.time()
    # train(document)
    # print("--- %s seconds ---" % (time.time() - start_time))
    # test()

main()

