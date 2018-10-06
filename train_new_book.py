from MongoDBTemplate import MongoDBTemplate
from doubanBook_config import *
from printer import predict_and_save

mongo = MongoDBTemplate(book_database_name, book_data_collection_name)

def train_new_book():
    books_collection=mongo.get_collection_by_collectionName('books')
    comment_collection=mongo.get_collection_by_collectionName('book_comment')
    data=books_collection.find({'machine_label':'0'},projection={'book_id':True})
    for item in data:
        book_id=item.get('book_id')
        print("train book id is {}".format(book_id))
        comment_data=comment_collection.find({'book_id':book_id})
        arr=[]
        for object in comment_data:
            arr.extend(object.get('original_book_comment').get('comments'))
        predict_and_save(book_id,arr)


if __name__ == '__main__':
    train_new_book()

