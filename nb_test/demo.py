import os
import jieba
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import time

"""
1.数据的预处理
"""


def preprocess(path):
    text_with_space = ""
    textfile = open(path, "r", encoding="utf8").read()
    textcute = jieba.cut(textfile)
    for word in textcute:
        text_with_space += word + " "
    return text_with_space


"""
2. 数据集分类标记
"""


def loadtrainset(path, classtag):
    allfiles = os.listdir(path)
    processed_textset = []
    allclasstags = []
    for thisfile in allfiles:
        # print(thisfile)
        path_name = path + "/" + thisfile
        processed_textset.append(preprocess(path_name))
        allclasstags.append(classtag)
    return processed_textset, allclasstags


processed_textdata1, class1 = loadtrainset("/Users/fengyang/PycharmProjects/NLP/dataset/train/hotel", "宾馆")
processed_textdata2, class2 = loadtrainset("/Users/fengyang/PycharmProjects/NLP/dataset/train/travel", "旅游")

train_data = processed_textdata1 + processed_textdata2
classtags_list = class1 + class2
# 对文本中的词语转换
count_vector = CountVectorizer()
vecot_matrix = count_vector.fit_transform(train_data)

"""
3. 特征提取与训练
"""
# TFIDF
# 提取特征
train_tfidf = TfidfTransformer(use_idf=False).fit_transform(vecot_matrix)
# 特征训练
clf = MultinomialNB().fit(train_tfidf, classtags_list)
"""
4. 测试
"""
testset = []

path = "/Users/fengyang/PycharmProjects/NLP/dataset/test/hotel"
allfiles = os.listdir(path)

hotel = 0
travel = 0

for thisfile in allfiles:
    path_name = path + "/" + thisfile
    new_count_vector = count_vector.transform([preprocess(path_name)])
    new_tfidf = TfidfTransformer(use_idf=False).fit_transform(new_count_vector)
    predict_result = clf.predict(new_tfidf)
    print(predict_result)
    print(thisfile)

    if (predict_result == "宾馆"):
        hotel += 1
    if (predict_result == "旅游"):
        travel += 1

print("宾馆" + str(hotel))
print("旅游" + str(travel))
