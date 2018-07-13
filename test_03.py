
import gensim

model = gensim.models.Word2Vec.load('save_model_03.txt')

# w1 = "blue"
# w2 = "house"
# print(model.wv.most_similar_cosmul(positive=[w1, w2], topn=10))

# w2 = "word2vec"
# print(model.wv.most_similar_cosmul(positive=w2, topn=10))
#
# w3 = "wrapsody"
# print(model.wv.most_similar_cosmul(positive=w3, topn=10))
#
# w4 = "sparrow"
# print(model.wv.most_similar_cosmul(positive=w4, topn=10))
#
# w5 = "fasoo"
# print(model.wv.most_similar_cosmul(positive=w5, topn=10))