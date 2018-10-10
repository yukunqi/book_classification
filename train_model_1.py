# -*- coding: utf-8 -*-

import re
import jieba.posseg as pseg
import logging
from time import time
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy import sparse
from MongoDBTemplate import MongoDBTemplate
from doubanBook_config import *
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import random
from gensim import models
from sklearn.model_selection import cross_val_score,cross_validate
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#迭代获得全部数据
def iter_get_all_data(collection_name,flag,batch_size=None):
    mongo = MongoDBTemplate(book_database_name, collection_name)
    collection = mongo.get_collection()
    comments_list = []
    y_train = []
    idList=collection.find({'machine_label':str(flag)},projection=['book_id'])
    count=0
    comment_collection=mongo.get_collection_by_collectionName('book_comment')
    for item in idList:
        current_id=item['book_id']
        print(current_id)
        com=comment_collection.find({'book_id':current_id},projection=['original_book_comment'],no_cursor_timeout=True)
        temp = ""
        for obj in com:
            comments=obj['original_book_comment']['comments']
            for comment in comments:
                line = keep_chinese(comment)
                content_list = pseg.cut(line)
                content = stopwords_filter(content_list)
                temp += content + ' '

        comments_list.append(temp)

        count += 1

        if batch_size is not None and count % batch_size == 0:
            for i in range(len(comments_list)):
                y_train.append(flag)
            yield comments_list, y_train
            comments_list = []
            y_train = []

    for i in range(len(comments_list)):
        y_train.append(flag)

    yield comments_list, y_train


#获取一定数量的数据
def get_data(collection_name,flag,skip_num=0,limit_num=0,filter=None):
    mongo = MongoDBTemplate(book_database_name, collection_name)
    collection = mongo.get_collection()
    comments_list = []
    y_train = []
    idList=collection.find({'machine_label':str(flag)},projection=['book_id']).skip(skip_num).limit(limit_num)
    comment_collection=mongo.get_collection_by_collectionName('book_comment')
    for item in idList:
        current_id=item['book_id']
        print(current_id)
        com=comment_collection.find({'book_id':current_id},projection=['original_book_comment'],no_cursor_timeout=True)
        temp = ""
        for obj in com:
            comments=obj['original_book_comment']['comments']
            for comment in comments:
                line = keep_chinese(comment)
                content_list = pseg.cut(line)
                content = stopwords_filter(content_list)
                temp += content + ' '

        comments_list.append(temp)

    for i in range(len(comments_list)):
        y_train.append(flag)

    return comments_list, y_train




def merge_data(dataList):
    data=[]
    y_train=[]
    for item in dataList:
       data.extend(item['data'])
       y_train.extend(item['target'])

    data_dict={}
    data_dict['data']=data
    data_dict['target']=y_train
    return data_dict

# 将原本一一对应的数据进行随机打乱重新组合 变成无序样本
def data_shuffle(item):
    data=[]
    target=[]
    c = list(zip(item['data'], item['target']))
    random.shuffle(c)
    data[:], target[:] = zip(*c)
    item['data']=data
    item['target']=target
    return item

# 为了防止训练集一次性加载过大 使用迭代器来分批提供训练集 做到 memory-friendly
def iter_minibatches(trainData,trainLabel,batch_size=None):
    X=[]
    y=[]
    cur_line_num=0
    data_len=len(trainData)
    print("data_len",data_len)
    for i in range(data_len):
        X.append(trainData[i])
        y.append(trainLabel[i])
        cur_line_num=cur_line_num+1
        if batch_size is not None and cur_line_num > batch_size:
            yield X,y
            X,y=[],[]
            cur_line_num=0
    yield X,y


#get the feature vector according to the dictionary 将特征列表转化成可以输入分类器的稀疏矩阵
def word2vec(words,dictionary):
    voc = dict(zip(dictionary,dictionary)) # 增加访问速度
    msgRow = []
    characterCol = []
    value = []

    print("word : {}".format(words))

    for i in range(len(words)):
         for word in words[i].split():
            if word in voc:
                msgRow.append(i)
                characterCol.append(dictionary.index(word))
                value.append(1)


    print(msgRow)
    print(characterCol)
    fea = sparse.coo_matrix((value,(msgRow,characterCol)),shape=(len(words),len(dictionary))).tocsr()
    return fea

# 过滤掉英文 字符 标点等符号 只保留中文字符
def keep_chinese(str):
    r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    obj = re.sub(r1,'',str)
    if obj:
        return obj
    else:
        return ""

# 停用词过滤
def stopwords_filter(list):
    try:
        stop_word = [(line.decode('utf-8').rstrip()) for line in open(stopwords_file_name, 'rb')]
        black_word = [(line.decode('utf-8').rstrip()) for line in open(blackwords_file_name, 'rb')]
        temp_list = []
        for item in list:
            if item.word not in stop_word and item.word not in black_word:
                temp_list.append(item.word)

        result = ' '.join(temp_list)
    except Exception as e:
        print(e)
        result=""

    return result

class MySentences(object):
    def __init__(self, wordList):
        self.wordList = wordList

    def __iter__(self):
        for line in self.wordList:
            yield line.split()

def iter_feature_selection_by_word2vecModel(model,trainData,positiveWordList,topN,useExistedModelPath=None):

    if useExistedModelPath is not None:
        model = models.Word2Vec.load(useExistedModelPath)
        vocab_len = len(model.wv.vocab)
        print("model vocab_len is ", vocab_len)
        value = model.most_similar(positive=positiveWordList, topn=topN - len(positiveWordList))
        dictionary = []
        for item in value:
            dictionary.append(item[0])

        dictionary.extend(positiveWordList)
        return dictionary

    sentences=MySentences(trainData)
    print(len(trainData))
    vocab_len = len(model.wv.vocab)
    print(sentences.wordList)
    print("model train finished vocab_len is ", vocab_len)
    if vocab_len == 0:
        print("build vocab")
        model.build_vocab(sentences=sentences,update=False)
    else:
        model.build_vocab(sentences=sentences, update=True)

    vocab_len = len(model.wv.vocab)
    print("model train finished vocab_len is ", vocab_len)
    model.train(sentences=sentences, total_examples=model.corpus_count, epochs=model.iter)
    #模型保存
    # model.save('./online/word2vec.model')
    value = model.most_similar(positive=positiveWordList, topn=topN - len(positiveWordList))
    dictionary = []
    for item in value:
        dictionary.append(item[0])
    dictionary.extend(positiveWordList)

    return dictionary


def iter_classifer():

    train_data_rumen=iter_get_all_data(collection_name='books',flag=1,batch_size=50)
    train_data_jinjie=iter_get_all_data(collection_name='books',flag=2,batch_size=20)

    rumen_stop_flag=0
    jinjie_stop_flag=0
    dic=[]
    classif=None
    testData_all=[]
    testLabel_all=[]
    model = models.Word2Vec(sentences=None, min_count=3, workers=20, size=400, batch_words=200)
    # clf = BernoulliNB(alpha=0.0001)
    # Here are some classifiers that support the `partial_fit` method
    partial_fit_classifiers = {
        'MultinomialNB': MultinomialNB(alpha=0.0001),
        'BernoulliNB':BernoulliNB(alpha=0.0001)
    }
    while True:
        rumen_data = None
        jinjie_data = None
        rumenLabel=None
        jinjieLabel=None
        if rumen_stop_flag == 0:
            try:
                rumen_data, rumenLabel = train_data_rumen.__next__()
            except StopIteration as e:
                print("入门数据用完")
                rumen_stop_flag=1

        if jinjie_stop_flag == 0:
            try:
                jinjie_data, jinjieLabel = train_data_jinjie.__next__()
            except StopIteration as e:
                print("进阶数据用完")
                jinjie_stop_flag=1


        if rumen_stop_flag==1 and jinjie_stop_flag == 1:
            print("所有数据使用完毕")
            return dic,classif

        dataList = []
        if rumen_data is not None and rumenLabel is not None:
            temp_dict={}
            temp_dict['data']=rumen_data
            temp_dict['target']=rumenLabel
            dataList.append(temp_dict)

        if jinjie_data is not None and jinjieLabel is not None:
            temp_dict = {}
            temp_dict['data'] = jinjie_data
            temp_dict['target'] = jinjieLabel
            dataList.append(temp_dict)


        total_data = merge_data(dataList)
        total_data = data_shuffle(total_data)  # 打乱数据样本 保证数据样本随机性

        trainData, testData, trainLabel, testLabel = train_test_split(total_data['data'], total_data['target'],
                                                                  test_size=0.4)

        if len(trainData) == 0:
            return dic, classif

        minibatch_test_iterators = iter_minibatches(trainData,trainLabel)

        testData_all.extend(testData)
        testLabel_all.extend(testLabel)
        print("测试集数据数量:  ",len(testData_all))
        print("测试集标签数量:  ",len(testLabel_all))

        #每次测试前都进行一次测试数据的随机打乱
        total_data['data']=testData_all
        total_data['target']=testLabel_all
        total_data=data_shuffle(total_data)
        testData_all,testLabel_all=total_data['data'],total_data['target']

        for i, (X_train, y_train) in enumerate(minibatch_test_iterators):
            dictionary = iter_feature_selection_by_word2vecModel(model, X_train, positiveWordList=['入门','初学者','通俗易懂','基础'], topN=200,useExistedModelPath='./online/word2vec.model')
            print("word2vec:", dictionary)
            print("word2vecLen: ", len(dictionary))

            dic=dictionary
            feaTest = word2vec(testData_all, dictionary)
            feaTrain = word2vec(X_train, dictionary)

            print("{} time".format(i))  # 当前次数

            for cls_name,clf in partial_fit_classifiers.items():
                print('_' * 80)
                print("Training: ")
                print(cls_name)
                clf.partial_fit(feaTrain,y_train,classes=[1,2])
                pred = clf.predict(feaTest)
                print('#' * 80)
                print(testLabel_all)
                print(pred)
                print('#' * 80)
                score = metrics.accuracy_score(testLabel_all, pred)
                print("accuracy:   %0.3f" % score)
                Fscore = metrics.f1_score(y_true=testLabel_all, y_pred=pred)
                print("F1score:   ", Fscore)
                Recall = metrics.recall_score(y_true=testLabel_all, y_pred=pred)
                print("Recall:  ", Recall)
                precision = metrics.precision_score(y_true=testLabel_all, y_pred=pred)
                print("Precision:  ", precision)
                if hasattr(clf, 'coef_'):
                    print("dimensionality: %d" % clf.coef_.shape[1])
                    print("density: %f" % density(clf.coef_))

                # clf_descr = str(clf).split('(')[0]
                # classif=clf
                # joblib.dump(clf, './online/' + clf_descr + '_model.pkl')


def build_word2vec_model():
    train_data_rumen = iter_get_all_data(collection_name='books', flag=1, batch_size=100)
    model = models.Word2Vec(sentences=None, min_count=3, workers=20, size=400, batch_words=200)
    for i,(x_train,y_train) in enumerate(train_data_rumen):
        sentences = MySentences(x_train)
        vocab_len = len(model.wv.vocab)
        print("model train finished vocab_len is ", vocab_len)
        if vocab_len == 0:
            model.build_vocab(sentences=sentences, update=False)
        else:
            model.build_vocab(sentences=sentences, update=True)
        model.train(sentences=sentences, total_examples=model.corpus_count, epochs=model.iter)

    model.save('./online/word2vec.model')


def cross_validation(total_example_num=500):
    train_data_rumen = iter_get_all_data(collection_name='books', flag=1, batch_size=50)
    train_data_jinjie = iter_get_all_data(collection_name='books', flag=2, batch_size=20)

    rumen_stop_flag = 0
    jinjie_stop_flag = 0
    dic = []
    classif = None
    testData_all = []
    testLabel_all = []
    dataList = []
    model = models.Word2Vec(sentences=None, min_count=3, workers=20, size=400, batch_words=200)
    # clf = BernoulliNB(alpha=0.0001)
    # Here are some classifiers that support the `partial_fit` method
    partial_fit_classifiers = {
        'MultinomialNB': MultinomialNB(alpha=0.0001),
        'BernoulliNB': BernoulliNB(alpha=0.0001)
    }

    while True:
        rumen_data = None
        jinjie_data = None
        rumenLabel=None
        jinjieLabel=None
        if rumen_stop_flag == 0:
            try:
                rumen_data, rumenLabel = train_data_rumen.__next__()
            except StopIteration as e:
                print("入门数据用完")
                rumen_stop_flag=1

        if jinjie_stop_flag == 0:
            try:
                jinjie_data, jinjieLabel = train_data_jinjie.__next__()
            except StopIteration as e:
                print("进阶数据用完")
                jinjie_stop_flag=1


        if rumen_stop_flag==1 and jinjie_stop_flag == 1:
            print("所有数据使用完毕")
            return dic,classif


        if rumen_data is not None and rumenLabel is not None:
            temp_dict={}
            temp_dict['data']=rumen_data
            temp_dict['target']=rumenLabel
            dataList.append(temp_dict)

        if jinjie_data is not None and jinjieLabel is not None:
            temp_dict = {}
            temp_dict['data'] = jinjie_data
            temp_dict['target'] = jinjieLabel
            dataList.append(temp_dict)



        
        total_data = merge_data(dataList)
        total_data = data_shuffle(total_data)  # 打乱数据样本 保证数据样本随机性
        trainData, testData, trainLabel, testLabel = train_test_split(total_data['data'], total_data['target'],
                                                                      test_size=0.4)


        if len(total_data) > total_example_num:
            print("validation finished total example num is {}".format(len(dataList)))
            return dic,classif


        minibatch_test_iterators = iter_minibatches(trainData, trainLabel)

        testData_all.extend(testData)
        testLabel_all.extend(testLabel)
        print("测试集数据数量:  ", len(testData_all))
        print("测试集标签数量:  ", len(testLabel_all))

        # 每次测试前都进行一次测试数据的随机打乱
        total_data['data'] = testData_all
        total_data['target'] = testLabel_all
        total_data = data_shuffle(total_data)
        testData_all, testLabel_all = total_data['data'], total_data['target']

        for i, (X_train, y_train) in enumerate(minibatch_test_iterators):
            dictionary = iter_feature_selection_by_word2vecModel(model, X_train,
                                                                 positiveWordList=['入门', '初学者', '通俗易懂', '基础'], topN=200,
                                                                 useExistedModelPath='./online/word2vec.model')
            print("word2vec:", dictionary)
            print("word2vecLen: ", len(dictionary))

            dic = dictionary
            feaTest = word2vec(testData_all, dictionary)
            feaTrain = word2vec(X_train, dictionary)

            print("{} time".format(i))  # 当前次数

            for cls_name, clf in partial_fit_classifiers.items():
                print('_' * 80)
                print("Training: ")
                print(cls_name)
                clf.partial_fit(feaTrain, y_train, classes=[1, 2])
                scoring=['precision_micro','recall_micro','f1_micro','accuracy']
                scores=cross_validate(clf,feaTrain,y_train,cv=5,scoring=scoring)
                print(scores.keys())
                print("train  accuracy: {}".format(scores['train_accuracy']))
                print("test  accuracy: {}".format(scores['test_accuracy']))
                print("train  f1:  {}".format(scores['train_f1_micro']))
                print("test  f1:  {}".format(scores['test_f1_micro']))



#利用PCA对数据进行降维可视化
def data_visualization_PCA(X,y):

    dictionary = iter_feature_selection_by_word2vecModel(None, None, positiveWordList=['入门', '初学者', '通俗易懂', '基础'],
                                                         topN=200, useExistedModelPath='./online/word2vec.model')

    fea=word2vec(X,dictionary)

    pca=PCA(n_components=2)
    reduce_X=pca.fit_transform(fea.todense())

    class_1_x,class_1_y=[],[]
    class_2_x,class_2_y=[],[]

    for i in range(len(reduce_X)):
        if y[i]==1:
            class_1_x.append(reduce_X[i][0])
            class_1_y.append(reduce_X[i][1])
        else:
            class_2_x.append(reduce_X[i][0])
            class_2_y.append(reduce_X[i][1])


    plt.scatter(class_1_x,class_1_y,c='r',marker='x')
    plt.scatter(class_2_x,class_2_y,c='b',marker='o')

    plt.show()

if __name__ == '__main__':
    #iter_classifer()
    #build_word2vec_model()
    #ross_validation()
    rumen_data,rumen_target=get_data(collection_name='books',flag=1,limit_num=4)
    jinjie_data,jinjie_target=get_data(collection_name='books',flag=2,limit_num=4)

    X,y=[],[]

    X.extend(rumen_data)
    y.extend(rumen_target)
    X.extend(jinjie_data)
    y.extend(jinjie_target)

    total={}
    total['data']=X
    total['target']=y



    total=data_shuffle(total)

    print("total data len : {}".format(len(total['data'])))


    data_visualization_PCA(total['data'],total['target'])