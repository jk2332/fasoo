import pickle
from os import listdir
import nltk
import gensim
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from konlpy.tag import Twitter
lm = WordNetLemmatizer()
twit = Twitter()
import logging
import urllib, json
from urllib.request import urlopen
from elasticsearch import Elasticsearch



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
    pickle.dump(idList, file)
    del file, idList


class Model:
    # directory for saving tokens of each document
    token_dir = "/Users/2018_in06/WikiExtractor_To_the_one_text/tokens/"

    # dictionary for checking pos tag
    part = {
        'N': 'n',
        'V': 'v',
        'J': 'a',
        'S': 's',
        'R': 'r'
    }

    def __init__(self, idList):
        self.model = None
        self.idList = idList
        self.documents = list(self.preprocess())

    # creating lemmatized tokens for a document with this id
    def new_tokenize(self, id):
        oriDoc = createJson(default+id)['_source']['content']
        temp=[]
        doc = twit.pos(oriDoc)
        if not oriDoc == "" and not doc[0][0].startswith("AAA"):
            for wandpos in doc:
                p = wandpos[1]
                word = wandpos[0]
                if p == "Alpha":
                    tag = nltk.pos_tag(word_tokenize(word))[0][1][0]
                    if tag in self.part.keys():
                        pos = self.part[tag]
                        temp.append(lm.lemmatize(word, pos))
                elif p == "Noun":
                    temp.append(word)
                if not temp:
                    continue
                yield temp

    # preprocess all the documents in idList and store tokens for each in the local
    def preprocess(self):
        for i, id in enumerate(self.idList):
            print("id: "+id)
            if (i % 100 == 0):
                logging.info("read {0} documents".format(i))

            # token file for doc with this id already exists
            if id+"_tokens.txt" in listdir(self.token_dir):
                print("************loading*************")
                file = open(self.token_dir+id+"_tokens.txt", 'rb')  # open file for reading
                g = lambda x: (n for n in x)
                try:
                    for line in g(pickle.load(file)):
                        yield line
                except:
                    # failed loading - maybe token file is empty
                    print("////////////// error occured: "+id)

            # create tokens for doc with this id
            else:
                print("************saving*************")
                file = open(self.token_dir+id+"_tokens.txt", "wb")  # open file for writing
                pickle.dump(list(self.new_tokenize(id)), file, protocol=pickle.HIGHEST_PROTOCOL)
            del file


    def create(self, sg=1, size=150, window=5, workers=3, batch_words=10000, min_count=5):
        self.model = gensim.models.Word2Vec(self.documents, sg, size, window, workers, batch_words, min_count)


    def train(self, epoch=15):
        if self.model is None:
            raise Exception('Model has not been created')
        self.model.train(self.documents, total_examples=self.model.corpus_count, epochs=epoch)



fname = "/Users/2018_in06/WikiExtractor_To_the_one_text/idlist/idlist10000.txt"
#saveIdList(fname)
file = open(fname, 'rb')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
m = Model(pickle.load(file)).create().train()
m.model.save("/Users/2018_in06/WikiExtractor_To_the_one_text/model/model10000tokenized.txt")
del m, file



