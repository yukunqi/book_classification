from D import insert_book_comments,mongo
from train_model import feature_selection_by_word2vecModel,word2vec,keep_chinese,stopwords_filter
from sklearn.externals import joblib
import threading
import jieba.posseg as pseg
import queue


class Printer(threading.Thread):
    def __init__(self, work_queue):
        super().__init__()  # 必须调用
        self.work_queue = work_queue

    def run(self):
        while True:
            _dict = self.work_queue.get()  # 当队列为空时，会阻塞，直到有数据
            _id=_dict['book_id']
            print("获取到项目任务 书籍ID为:{} 当前任务队列大小为 {}".format(_id, work_queue.qsize()))
            comment_arr = insert_book_comments(collection_name='book_comment', book_id=_id, sleep_time=2)
            predict_and_save(_id,comment_arr)
            print("书籍ID为 {} 的分类和数据存储任务完成 当前任务队列大小为 {}".format(_id,work_queue.qsize()))
            self.work_queue.task_done()


def predict_and_save(book_id,comment_arr):

    dictonary=feature_selection_by_word2vecModel(trainData=None,positiveWordList=['入门','初学者','通俗易懂','基础'],topN=200,useExistedModelPath='./online/word2vec_once.model')
    #print(dictonary)
    comment_arr=data_fenci(comment_arr)
    feaTest=word2vec(comment_arr,dictonary)
    try:
        clf_muti=joblib.load('./online/MultinomialNB_model.pkl')
        pred_label=clf_muti.predict(feaTest)
        if pred_label[0] == 1 or pred_label[0] == 2:
            print("预测的标签为 {} ".format(pred_label[0]))
            books_collection=mongo.get_collection_by_collectionName('books')
            books_collection.update_one({'book_id':book_id},{'$set':{'machine_label':str(pred_label[0])}})
        else:
            raise Exception("预测标签异常 预测出的标签出现规定类别范围之外 标签为{}".format(pred_label))
        return True
    except FileNotFoundError as e:
        print(e)
        return False


def data_fenci(data_arr):
    comments_list=[]
    temp=""
    for index in range(len(data_arr)):

        line = data_arr[index]

        line = keep_chinese(line)

        content_list = pseg.cut(line)

        content = stopwords_filter(content_list)

        temp += content + ' '
    comments_list.append(temp)
    return comments_list


work_queue = queue.Queue()

