# coding: utf-8
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
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from apscheduler.schedulers.blocking import BlockingScheduler
import sys
import jieba
import bson

# 这里写你自己的框架保存地址
sys.path.append('F:\PycharmProjects\DEMO1\IP_POOL')
sys.path.append('F:/PycharmProjects/DEMO1/AiSpider')

#jieba.enable_parallel()
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,filename='log.txt')



# 从数据库中读取相应collection的一定数量的评论数据 并打上flag标签标记成一类
def get_data(collection_name,flag,skip_num=0,limit_num=0,filter=None):
    mongo = MongoDBTemplate(book_database_name, collection_name)
    collection = mongo.get_collection()
    pipeline=[{'$lookup':
        {
            'from':'book_comment',
            'localField':'book_id',
            'foreignField':'book_id',
            'as':'book_comment_docs'
        }# 联合查询 把collectionName的表和book_comment表联合查找相应书籍的评论数据
    },
        {'$limit':limit_num},# 限制返回数量
        {'$skip':skip_num}, #跳过结果的数量
        #{'$match':filter},# 根据过滤条件查找指定数据
        {'$project':{'book_id':1,'book_comment_docs.original_book_comment':1}} #指定需要返回的字段
    ]
    data = collection.aggregate(pipeline=pipeline)
    comments_list = []
    y_train = []
    book_idList=[]
    for item in data:
        print(item['book_id'])
        book_idList.append(item['book_id'])
        for comment in item['book_comment_docs']:
            list=comment['original_book_comment']['comments']
            temp=""
            for index in range(len(list)):
                line = list[index]
                line = keep_chinese(line)
                content_list = pseg.cut(line)
                content = stopwords_filter(content_list)
                temp+=content+' '
            comments_list.append(temp)

    for i in range(len(comments_list)):
        y_train.append(flag)

    dict_data={}
    dict_data['data']=comments_list
    dict_data['target']=y_train
    dict_data['idList']=book_idList

    return dict_data

def get_new_data(collection_name,flag,newestID=None,skip_num=0,filter=None):
    mongo = MongoDBTemplate(book_database_name, collection_name)
    collection = mongo.get_collection()
    pipeline=[{'$lookup':
        {
            'from':'book_comment',
            'localField':'book_id',
            'foreignField':'book_id',
            'as':'book_comment_docs'
        }# 联合查询 把collectionName的表和book_comment表联合查找相应书籍的评论数据
    },
        {'$skip':skip_num}, #跳过结果的数量
        {'$match':filter},# 根据过滤条件查找指定数据
        {'$project':{'book_id':1,'book_comment_docs.original_book_comment':1}} #指定需要返回的字段
    ]
    data = collection.aggregate(pipeline=pipeline)
    comments_list = []
    y_train = []
    book_idList=[]
    tempID=newestID
    for item in data:
        print(item['book_id'])
        book_idList.append(item['book_id'])
        tempID=item['_id']
        for comment in item['book_comment_docs']:
            list=comment['original_book_comment']['comments']
            temp=""
            for index in range(len(list)):
                line = list[index]
                line = keep_chinese(line)
                content_list = pseg.cut(line)
                content = stopwords_filter(content_list)
                temp+=content+' '
            comments_list.append(temp)

    for i in range(len(comments_list)):
        y_train.append(flag)

    dict_data={}
    dict_data['data']=comments_list
    dict_data['target']=y_train
    dict_data['newest_id']=tempID
    dict_data['idList']=book_idList

    return dict_data


def iter_get_data_with_pageLimit(collection_name,flag,start_page,end_page,pageSize):
    mongo = MongoDBTemplate(book_database_name, collection_name)
    collection = mongo.get_collection()

    comments_list = []
    y_train = []

    while start_page <= end_page:
        pipeline = [{'$lookup':
            {
                'from': 'book_comment',
                'localField': 'book_id',
                'foreignField': 'book_id',
                'as': 'book_comment_docs'
            }  # 联合查询 把collectionName的表和book_comment表联合查找相应书籍的评论数据
        },
            {'$limit': pageSize},  # 限制返回数量
            {'$skip': (start_page-1)*pageSize},  # 跳过结果的数量
            {'$project': {'book_id': 1, 'book_comment_docs.original_book_comment': 1}}  # 指定需要返回的字段
        ]
        data = collection.aggregate(pipeline=pipeline)
        print("page:  ",start_page)
        for item in data:
            print(item['book_id'])
            for comment in item['book_comment_docs']:
                list = comment['original_book_comment']['comments']
                temp = ""
                for index in range(len(list)):
                    line = list[index]
                    line = keep_chinese(line)
                    content_list = pseg.cut(line)
                    content = stopwords_filter(content_list)
                    temp += content + ' '
                comments_list.append(temp)

        for i in range(len(comments_list)):
            y_train.append(flag)

        yield comments_list, y_train

        comments_list=[]
        y_train=[]

        start_page+=1




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

def test_read_data_with_sql():
    train_data_rumen=get_data('book_rumen',limit_num=400,flag=1)
    train_data_jinjie=get_data('book_jinjie',limit_num=400,flag=2)

    dataList=[]
    dataList.append(train_data_rumen)
    dataList.append(train_data_jinjie)
    total_data=merge_data(dataList)
    total_data=data_shuffle(total_data) #打乱数据样本 保证数据样本随机性

    trainData,testData,trainLabel,testLabel=train_test_split(total_data['data'],total_data['target'],test_size=0.4)

    classifier(trainData,testData,trainLabel,testLabel)
    #Search_for_best(trainData,trainLabel)


def svmClassifer(feaTrain, trainLabel, feaTest,testLabel):
    clf = LinearSVC(C=0.8)
    benchmark(clf,feaTrain,trainLabel,feaTest,testLabel)



def logisticReg(feaTrain, trainLabel, feaTest,testLabel):
    clf = LogisticRegression()
    benchmark(clf,feaTrain,trainLabel,feaTest,testLabel)


#get the feature vector according to the dictionary 将特征列表转化成可以输入分类器的稀疏矩阵
def word2vec(words,dictionary):
    voc = dict(zip(dictionary,dictionary)) # 增加访问速度
    msgRow = []
    characterCol = []
    value = []
    for i in range(len(words)):
         for word in words[i].split():
            if word in voc:
                msgRow.append(i)
                characterCol.append(dictionary.index(word))
                value.append(1)


    fea = sparse.coo_matrix((value,(msgRow,characterCol)),shape=(len(words),len(dictionary))).tocsr()
    return fea


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
    vocab_len = len(model.wv.vocab)
    print("model train finished vocab_len is ", vocab_len)
    if vocab_len == 0:
        model.build_vocab(sentences=sentences,update=False)
    else:
        model.build_vocab(sentences=sentences, update=True)

    model.train(sentences=sentences, total_examples=model.corpus_count, epochs=model.iter)
    #模型保存
    # model.save('./online/word2vec.model')
    value = model.most_similar(positive=positiveWordList, topn=topN - len(positiveWordList))
    dictionary = []
    for item in value:
        dictionary.append(item[0])
    dictionary.extend(positiveWordList)

    return dictionary

# 通过词向量进行特征选择 根据输入的训练集 进行模型训练 然后返回指定特征的相关特征 达到特征选择的目的
def feature_selection_by_word2vecModel(trainData,positiveWordList,topN,useExistedModelPath=None):

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
    model = models.Word2Vec(sentences=sentences, min_count=10, workers=20, size=400, batch_words=200)
    vocab_len = len(model.wv.vocab)
    print("model train finished vocab_len is ",vocab_len)
    # 保存模型
    model.save('./online/word2vec_once_with_reviews_10_mincount.model')
    value = model.most_similar(positive=positiveWordList,topn=topN-len(positiveWordList))
    dictionary=[]
    for item in value:
        dictionary.append(item[0])

    dictionary.extend(positiveWordList)

    #model.save('./cache/word2vec_' + str(vocab_len) + '.model')
    #model.wv.save_word2vec_format('./cache/word2vec_' + str(vocab_len) + '.bin', binary=True)

    return dictionary

# 通过网格搜索来自动优化参数 ，机器遍历给定参数的范围 然后给出分数最高的
def Search_for_best(trainData,trainLabel):
    pipeline = Pipeline([
        ('clf', BernoulliNB())
    ])
    parameters = {
        'clf__alpha': [0.0001,0.00001],
    }
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
    grid_search.fit(trainData,trainLabel)
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

def classifier(trainData, testData, trainLabel, testLabel):

    dictionary=feature_selection_by_word2vecModel(trainData,positiveWordList=['入门','初学者','通俗易懂','基础'],topN=200,useExistedModelPath='./online/word2vec_once.model')
    print("word2vec:",dictionary)
    feaTrain = word2vec(trainData, dictionary)
    feaTest = word2vec(testData, dictionary)
    #Search_for_best(feaTrain,trainLabel)


    logisticReg(feaTrain, trainLabel, feaTest, testLabel)
    svmClassifer(feaTrain, trainLabel, feaTest, testLabel)
    benchmark(MultinomialNB(alpha=0.0001), feaTrain, trainLabel, feaTest, testLabel)
    benchmark(BernoulliNB(alpha=0.0001), feaTrain, trainLabel, feaTest, testLabel)


def iter_classifer():

    train_data_rumen=iter_get_all_data(collection_name='book_rumen',flag=1,batch_size=50)
    train_data_jinjie=iter_get_all_data(collection_name='book_jinjie',flag=2,batch_size=20)

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
            dictionary = iter_feature_selection_by_word2vecModel(model, X_train, positiveWordList=['入门','初学者','通俗易懂','基础'], topN=200,useExistedModelPath='./online/word2vec_once.model')
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

# 分类函数 clf为分类器实例 训练集 训练标签 测试集 测试标签
def benchmark(clf,trainData,trainLabel,testData,testLabel,if_save_model=False):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(trainData, trainLabel)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(testData)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)
    score = metrics.accuracy_score(testLabel, pred)
    print("accuracy:   %0.3f" % score)
    Fscore=metrics.f1_score(y_true=testLabel,y_pred=pred)
    print("F1score:   ", Fscore)
    Recall=metrics.recall_score(y_true=testLabel,y_pred=pred)
    print("Recall:  ", Recall)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

    print()
    clf_descr = str(clf).split('(')[0]

    if if_save_model:
        print("save model: \n")
        joblib.dump(clf, './online/' + clf_descr + '_model.pkl')

    return clf_descr, score, train_time, test_time, pred


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


# 过滤掉英文 字符 标点等符号 只保留中文字符
def keep_chinese(str):
    try:
        r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
        obj = re.sub(r1,'',str)
        if obj:
            return obj
        else:
            return ""
    except Exception as e:
        print("error occured in keep chinese : {} \n and which str is {}".format(e,str))
        return ""

def find_latestID(collection_name):
    mongo = MongoDBTemplate(book_database_name, collection_name)
    collection = mongo.get_collection()
    data = collection.find().sort('_id',-1).limit(1)
    newest_objectID=data[0]['_id']
    return newest_objectID

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
        print("into exception")
        result=""

    #print("stop word result:{}".format(result))
    return result


def check_and_update_model():

    global newest_objectID_jinjie, newest_objectID_rumen

    logging.info("检测并更新模型程序开始执行")
    print("检测并更新模型程序开始执行")

    # 查找比上一次的ID更新的数据　训练最新数据并更新模型　保存最新ID
    data_rumen=get_new_data('book_rumen',flag=0,filter={'_id': {'$gt': newest_objectID_rumen}},newestID=newest_objectID_rumen)
    data_jinjie=get_new_data('book_jinjie', flag=1, filter={'_id': {'$gt': newest_objectID_jinjie}},newestID=newest_objectID_jinjie)

    # 没有新增数据进行训练 等待下一次执行
    if len(data_rumen['data']) == 0 and len(data_jinjie['data']) == 0:
        logging.info("检测完毕，没有新增数据 等待下一执行周期")
        print("检测完毕，没有新增数据 等待下一执行周期")
        return

    dataList = []
    dataList.append(data_rumen)
    dataList.append(data_jinjie)

    total_data = merge_data(dataList)
    total_data = data_shuffle(total_data)  # 打乱数据样本 保证数据样本随机性

    testData_all=[]
    testLabel_all=[]
    trainData, testData, trainLabel, testLabel = train_test_split(total_data['data'], total_data['target'],
                                                                  test_size=0.4)
    logging.info("获取到最新数据,开始训练")
    dictionary,clf=online_training(trainData,trainLabel)

    test=getTestData(test_size=0.4)

    # 测试集由两部分组成 一部分是新增数据中拿取部分，一部分是存量数据中拿取一部分
    testData_all.extend(testData)
    testData_all.extend(test['data'])
    testLabel_all.extend(testLabel)
    testLabel_all.extend(test['target'])

    online_testing(dictionary,clf,testData_all,testLabel_all)

    newest_objectID_rumen=data_rumen['newest_id']
    newest_objectID_jinjie=data_jinjie['newest_id']

#迭代获得全部数据
def iter_get_all_data(collection_name,flag,batch_size=None):
    mongo = MongoDBTemplate(book_database_name, collection_name)
    collection = mongo.get_collection()
    comments_list = []
    y_train = []
    pipeline=[{'$lookup':
        {
            'from':'book_comment',
            'localField':'book_id',
            'foreignField':'book_id',
            'as':'book_comment_docs'
        }# 联合查询 把collectionName的表和book_comment表联合查找相应书籍的评论数据
    },
        {'$project':{'book_id':1,'book_comment_docs.original_book_comment':1}} #指定需要返回的字段
    ]
    data = collection.aggregate(pipeline=pipeline,maxTimeMS=1200000)
    count=0
    for item in data:
        print(item['book_id'])
        for comment in item['book_comment_docs']:
            list=comment['original_book_comment']['comments']
            temp=""
            for index in range(len(list)):
                line = list[index]
                line = keep_chinese(line)
                content_list = pseg.cut(line)
                content = stopwords_filter(content_list)
                temp+=content+' '
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



def getTestData(test_size=0.5):
    mongo=MongoDBTemplate(database_name=book_database_name,collection_name='book_rumen')
    rumen_len=mongo.get_collection().count()
    jinjie_len=mongo.get_collection_by_collectionName('book_jinjie').count()

    rumen_limit=int(rumen_len*test_size)
    jinjie_limit=int(jinjie_len*test_size)

    data_rumen=get_data('book_rumen',flag=1,limit_num=rumen_limit)
    data_jinjie=get_data('book_jinjie',flag=2,limit_num=jinjie_limit)

    dataList=[]
    dataList.append(data_rumen)
    dataList.append(data_jinjie)
    total=merge_data(dataList)# 合并两个类别数据
    total=data_shuffle(total)# 打乱数据 保持样本随机性

    return total

def online_training(trainData,trainLabel):

    #加载之前训练好的word2vec模型和分类器 若不存在 则执行初始化训练word2vec和分类器 并保存在指定文件夹
    try:
        model = models.Word2Vec.load('./online/word2vec.model')
        clf = joblib.load('./online/BernoulliNB_model.pkl')
    except FileNotFoundError as e:
        logging.info(" word2vec模型或者分类器不存在 初始化训练")
        dictionary,clf=iter_classifer()
        return dictionary,clf

    minibatch_test_iterators = iter_minibatches(trainData, trainLabel)
    dic=[]
    classif=None

    partial_fit_classifiers=[]
    partial_fit_classifiers.append(clf)

    for i, (X_train, y_train) in enumerate(minibatch_test_iterators):
        dictionary = iter_feature_selection_by_word2vecModel(model, X_train, positiveWordList=['入门'], topN=200)
        print("word2vec:", dictionary)
        dic=dictionary
        feaTrain = word2vec(X_train, dictionary)
        print("{} time".format(i))  # 当前次数
        for clf in partial_fit_classifiers:
            print('_' * 80)
            print("Training: ")
            print(clf)
            clf.partial_fit(feaTrain, y_train, classes=[0, 1])
            clf_descr = str(clf).split('(')[0]
            classif=clf
            joblib.dump(clf, './online/' + clf_descr + '_model.pkl')

    return dic,classif


def online_testing(dictionary, clf, testData, testLabel):
    feaTest = word2vec(testData, dictionary)
    print('_' * 80)
    print("Training: ")
    print(clf)
    pred = clf.predict(feaTest)
    print('#' * 80)
    print(testLabel)
    print(pred)
    print('#' * 80)
    score = metrics.accuracy_score(testLabel, pred)
    print("accuracy:   %0.3f" % score)
    Fscore = metrics.f1_score(y_true=testLabel, y_pred=pred)
    print("F1score:   ", Fscore)
    Recall = metrics.recall_score(y_true=testLabel, y_pred=pred)
    print("Recall:  ", Recall)
    precision=metrics.precision_score(y_true=testLabel,y_pred=pred)
    print("Precision:  ",precision)
    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

def getReviewByBookID(collection_name,filter):
    mongo = MongoDBTemplate(book_database_name, collection_name)
    collection = mongo.get_collection()
    pipeline=[{'$lookup':
        {
            'from':'book_review',
            'localField':'book_id',
            'foreignField':'book_id',
            'as':'book_review_docs'
        }# 联合查询 把collectionName的表和book_comment表联合查找相应书籍的评论数据
    },
        {'$match':filter},
        {'$project':{'book_id':1,'book_review_docs.original_book_reviews':1}} #指定需要返回的字段
    ]
    data = collection.aggregate(pipeline=pipeline)
    reviews_list = []

    for item in data:
        print(item['book_id'])
        for comment in item['book_review_docs']:
            list=comment['original_book_reviews']['reviews']
            temp=""
            for index in range(len(list)):
                line = list[index]
                line = keep_chinese(line)
                content_list = pseg.cut(line)
                content = stopwords_filter(content_list)
                temp+=content+' '
            reviews_list.append(temp)

    dict_data={}
    dict_data['data']=reviews_list
    return dict_data

def getDataByBookID(collection_name,filter):

    mongo = MongoDBTemplate(book_database_name, collection_name)
    collection = mongo.get_collection()
    pipeline=[{'$lookup':
        {
            'from':'book_comment',
            'localField':'book_id',
            'foreignField':'book_id',
            'as':'book_comment_docs'
        }# 联合查询 把collectionName的表和book_comment表联合查找相应书籍的评论数据
    },
        {'$match':filter},
        {'$project':{'book_id':1,'book_comment_docs.original_book_comment':1}} #指定需要返回的字段
    ]
    data = collection.aggregate(pipeline=pipeline)
    comments_list = []

    for item in data:
        print(item['book_id'])
        for comment in item['book_comment_docs']:
            list=comment['original_book_comment']['comments']
            temp=""
            for index in range(len(list)):
                line = list[index]
                line = keep_chinese(line)
                content_list = pseg.cut(line)
                content = stopwords_filter(content_list)
                temp+=content+' '
            comments_list.append(temp)

    dict_data={}
    dict_data['data']=comments_list
    return dict_data

# 根据模型的预测数据去更新训练模型，测试模型的指标
def test_model_train_byPredictData():
    # 首先测试模型的性能
    test_data_rumen=get_data('book_rumen',flag=1,skip_num=380,limit_num=120)
    test_dat_jinjie=get_data('book_jinjie',flag=2,skip_num=380,limit_num=120)

    dataList = []
    dataList.append(test_data_rumen)
    dataList.append(test_dat_jinjie)
    total_data = merge_data(dataList)
    total_data = data_shuffle(total_data)  # 打乱数据样本 保证数据样本随机性
    dictonary=feature_selection_by_word2vecModel(trainData=None,positiveWordList=['入门','初学者','通俗易懂','基础'],topN=200,useExistedModelPath='./online/word2vec_once.model')

    try:
        clf_Ber=joblib.load('./online/BernoulliNB_model.pkl')
        clf_muti=joblib.load('./online/MultinomialNB_model.pkl')
        online_testing(dictonary,clf_Ber,total_data['data'],total_data['target'])
        online_testing(dictonary, clf_muti, total_data['data'], total_data['target'])

        # 利用部分数据进行预测，将预测结果再训练更新模型
        test_data_rumen = get_data('book_rumen', flag=1, skip_num=300, limit_num=80)
        test_dat_jinjie = get_data('book_jinjie', flag=2, skip_num=300, limit_num=80)
        dataList = []
        dataList.append(test_data_rumen)
        dataList.append(test_dat_jinjie)
        total_data1 = merge_data(dataList)
        total_data1 = data_shuffle(total_data1)  # 打乱数据样本 保证数据样本随机性
        feaTest = word2vec(total_data1['data'], dictonary)

        arr1=clf_muti.predict(feaTest)

        # 更新样本标签
        for index in range(len(arr1)):

            if total_data1['target'][index]!=arr1[index]:
                print('original label:{} predict label: {} bookId : {}'.format(total_data1['target'][index],arr1[index],total_data1['idList'][index]))
            total_data1['target'][index]=arr1[index]

        #增量学习
        clf_muti.partial_fit(feaTest, total_data1['target'], classes=[1,2])

        online_testing(dictonary,clf_muti,total_data['data'],total_data['target'])

    except FileNotFoundError as e:
        print(e)

def move_data():
    mongo = MongoDBTemplate(book_database_name,'book_jinjie')
    mongo_comment=mongo.get_collection_by_collectionName('book_review')

    collection = mongo.get_collection()
    data = collection.find(filter=None,skip=0,projection=['book_id','reviews'],no_cursor_timeout=True)


    for item in data:
        tempID=item['book_id']
        print(tempID)
        list = item['reviews']
        bson={'book_id':tempID,'original_book_reviews':list}
        mongo_comment.insert(bson)
        collection.update_one(filter={'book_id':tempID},update={'$unset':{'reviews':'','review_total_count':''}})




if __name__ == '__main__':

    #test_read_data_with_sql() # 一次性训练
    iter_classifer()# 增量迭代训练
    # test_model_train_byPredictData()

    # newest_objectID_rumen=find_latestID('book_rumen')
    # newest_objectID_jinjie=find_latestID('book_jinjie')
    #
    # scheduler = BlockingScheduler()
    # scheduler.add_job(check_and_update_model, 'interval', minutes=10)
    # scheduler.start()

    #book_data=getDataByBookID('book_rumen',filter={'book_id':'1045818'})

    # 计算一个document的占用空间大小
    # encoded=bson.BSON.encode(document=book_data)
    # max_bson_size=len(encoded)
    # print(' %d bytes' % max_bson_size)

    # dictonary=feature_selection_by_word2vecModel(trainData=None,positiveWordList=['入门','初学者','通俗易懂','基础'],topN=200,useExistedModelPath='./online/word2vec_once.model')
    # print(dictonary)
    # feaTest=word2vec(book_data['data'],dictonary)
    # try:
    #     clf_Ber=joblib.load('./online/BernoulliNB_model.pkl')
    #     clf_muti=joblib.load('./online/MultinomialNB_model.pkl')
    #
    #     print(clf_Ber.predict(feaTest))
    #     print(clf_muti.predict(feaTest))
    # except FileNotFoundError as e:
    #     print(e)
    #get_data('book_rumen',flag=1,skip_num=0,limit_num=10)

