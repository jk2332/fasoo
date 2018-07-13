import pickle
import nltk
import gensim
nltk.download('wordnet')
import logging
import urllib, json
from urllib.request import urlopen


default = "http://192.168.101.183:9200/wrapsody/document/"
def createJson(url):
    with urllib.request.urlopen(url) as url2:
        data = json.loads(url2.read().decode('utf-8'))
        return data


def saveIdList(fname):
    dic = createJson(default + "_search?size=10000&_source=false&sort=_id:desc")
    idList = []
    for e in dic["hits"]["hits"]:
        id = e['_id']
        if id[0].isdigit():
            idList.append(id)
    file = open(fname, "wb")
    pickle.dump(idList, file, protocol=pickle.HIGHEST_PROTOCOL)


def read_input(idList):
    for i, id in enumerate(idList):
        oriDoc = createJson(default + id)['_source']['content']
        print("id: " + id)
        if (i % 100 == 0):
            logging.info("read {0} documents".format(i))
        yield gensim.utils.simple_preprocess(oriDoc)


fname = "/Users/2018_in06/WikiExtractor_To_the_one_text/idlist/idlist50000.txt"
docdir = "/Users/2018_in06/WikiExtractor_To_the_one_text/documents/10000docv1.txt"

saveIdList(fname)
file = open(fname, 'rb')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
documents = list(read_input(pickle.load(file)))

file = open(docdir, 'wb')
pickle.dump(documents, file, protocol=pickle.HIGHEST_PROTOCOL)


model = gensim.models.Word2Vec(documents, sg=1, size=150, window=5, workers=3, batch_words=10000, min_count=5)
model.train(documents, total_examples=len(documents), epochs=15)

model.save("/Users/2018_in06/WikiExtractor_To_the_one_text/model/model10000.txt")