# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import requests
import re
import json
import time
from IP_List import IPs
from proxy_basic_config import collection_name
from AiSpider.spider.log_format import spider_log
from AiSpider.spider.UAS import PC_USER_AGENTS
from apscheduler.schedulers.background import BackgroundScheduler
import random
from doubanBook_config import *
from MongoDBTemplate import MongoDBTemplate


url_list = ["https://book.douban.com/tag/%E7%BC%96%E7%A8%8B?start={}&type=T".format(str(i)) for i in range(0, 1000, 20)]
#初始化IP集合对象
ips = IPs(collection_name)
#当前Python文件的logger对象
logger = spider_log(log_name="getbook")
mongo = MongoDBTemplate(book_database_name, book_data_collection_name)


# 获取一个标签下的所有书籍的数据
def get_books_data(tag_str, page=1,exclude_collection=None,page_limit=None):
    pageSize = 20
    res = []
    book_id_list=[]

    for name in exclude_collection:
        coll = mongo.get_collection_by_collectionName(name)
        record = coll.find({}, projection={'book_id': True, '_id': False})
        for it in record:
            book_id_list.append(it['book_id'])

    while True:
        url = "https://book.douban.com/tag/" + tag_str + "?start=" + str((page - 1) * pageSize)
        web_data = request_with_proxy(url,True)
        soup = BeautifulSoup(web_data.text, 'lxml')
        items=soup.select('#subject_list > ul > li.subject-item')
        for it in items:
            rate=it.find('span',class_='rating_nums')
            book_url=it.find('a').get('href')
            book_id = get_book_id_from_url(book_url)
            if rate and float(rate.get_text()) >= 7 and book_id not in book_id_list:
                msg="ready to get book book_id is {}  rate is {}".format(book_id,rate.get_text())
                logger.info(msg)
                res.append(get_book_data_byID(book_id))
        if len(items) == 0 or (page_limit is not None and page >= page_limit):
            break

        page = page + 1




# 获取指定bookID的短评页面
def get_book_comments(book_id=None, page=1,page_limit=None, sleep_time=2):
    list = []
    total_comment_count = ""
    while True:
        try:
            url = "https://book.douban.com/subject/" + book_id + "/comments/hot?p=" + str(page)
            web_data = request_with_proxy(url, True)
            msg = "success catch book_comment url is {} current page is {}".format(url,page)
            logger.info(msg)
            soup = BeautifulSoup(web_data.text, 'lxml')
            commments = soup.select('#comments > ul > li > div.comment > p.comment-content')
            #total_comment_count = soup.select("#total-comments")
            temp=[]
            for comment in commments:
                temp.append(comment.get_text())


            for item in temp:
                list.append(item)
            if len(temp) == 0 or (page_limit is not None and page >= page_limit):
                break

            page = page + 1
            time.sleep(sleep_time)
        except requests.exceptions.ProxyError as e:
            logger.info(e)
            logger.info("代理异常.... 返回部分数据  " + "  当前页数为...", page)
            break
        except Exception as e:
            logger.info(e)
            logger.info("发生未知异常，异常发生页面URL为:", url)
            break

    dict = {}
    dict['total_comment_count'] =len(list)
    dict['comments'] = list
    return dict


# 根据正则去匹配URL中获取书籍ID
def get_book_id_from_url(url):
    obj = re.match('^(f|ht){1}(tp|tps):\/\/book.douban.com[^\d]*(\d+)', url)
    if obj:
        try:
            return obj.group(3)
        except IndexError as e:
            print('except: URL中匹配不到指定的书籍ID　URL有误')
            return -1
    else:
        print(url + "不是我们正常需要的URL　正确例子：https://book.douban.com/subject/27119594/")


# 通过API根据书籍id去获取相应的标签信息
def get_tags_from_api(book_id):
    url = "https://api.douban.com/v2/book/" + book_id + "/tags"
    web_data = request_with_proxy(url, True)
    # str='{"count":50,"start":0,"total":188,"tags":[{"count":1574,"name":"算法","title":"算法"},{"count":638,"name":"计算机","title":"计算机"},{"count":561,"name":"编程","title":"编程"},{"count":295,"name":"计算机科学","title":"计算机科学"},{"count":286,"name":"Algorithms","title":"Algorithms"},{"count":266,"name":"计算机-算法","title":"计算机-算法"},{"count":179,"name":"经典","title":"经典"},{"count":154,"name":"图灵程序设计丛书","title":"图灵程序设计丛书"},{"count":137,"name":"Java","title":"Java"},{"count":120,"name":"数据结构","title":"数据结构"},{"count":95,"name":"基础理论","title":"基础理论"},{"count":95,"name":"programming","title":"programming"},{"count":58,"name":"程序设计","title":"程序设计"},{"count":49,"name":"IT","title":"IT"},{"count":41,"name":"java","title":"java"},{"count":15,"name":"软件开发","title":"软件开发"},{"count":15,"name":"Programming","title":"Programming"},{"count":13,"name":"技术","title":"技术"},{"count":11,"name":"程序员","title":"程序员"},{"count":11,"name":"互联网","title":"互联网"},{"count":9,"name":"数据结构与算法","title":"数据结构与算法"},{"count":9,"name":"数学","title":"数学"},{"count":8,"name":"科普","title":"科普"},{"count":8,"name":"CS","title":"CS"},{"count":8,"name":"Algorithm","title":"Algorithm"},{"count":6,"name":"算法、数据结构","title":"算法、数据结构"},{"count":4,"name":"入门","title":"入门"},{"count":3,"name":"科学","title":"科学"},{"count":3,"name":"数据结构与算法分析","title":"数据结构与算法分析"},{"count":2,"name":"软件工程","title":"软件工程"},{"count":2,"name":"计算机算法","title":"计算机算法"},{"count":2,"name":"計算機","title":"計算機"},{"count":2,"name":"美国","title":"美国"},{"count":2,"name":"编程语言","title":"编程语言"},{"count":2,"name":"算法与数据结构","title":"算法与数据结构"},{"count":2,"name":"算法,计算机,编程","title":"算法,计算机,编程"},{"count":2,"name":"数据分析","title":"数据分析"},{"count":2,"name":"教材","title":"教材"},{"count":2,"name":"思维","title":"思维"},{"count":2,"name":"学习","title":"学习"},{"count":2,"name":"外国文学","title":"外国文学"},{"count":2,"name":"java算法","title":"java算法"},{"count":2,"name":"algorithm","title":"algorithm"},{"count":2,"name":"4","title":"4"},{"count":2,"name":"0","title":"0"},{"count":1,"name":"黑客","title":"黑客"},{"count":1,"name":"随便看看","title":"随便看看"},{"count":1,"name":"重要度.\/.××","title":"重要度.\/.××"},{"count":1,"name":"购书单","title":"购书单"},{"count":1,"name":"设计","title":"设计"}]}'
    json_dict = json.loads(web_data.text)
    return json_dict['tags']


# 根据书籍页面去获取相应的标签信息
def get_tags_from_page(book_id):
    url = "https://book.douban.com/subject/" + book_id + "/"
    web_data=request_with_proxy(url,True)
    soup = BeautifulSoup(web_data.text, 'lxml')
    tags = soup.select("#db-tags-section > div.indent")
    return list(tags[0].stripped_strings)


# 分页抓取一本书所有书评页面数据
def get_book_reviews(book_id, page=1,page_limit=None, sleep_time=2):
    list = []
    pageSize = 20
    total_review_count=0
    while True:
        try:
            url = "https://book.douban.com/subject/" + book_id + "/reviews?start=" + str((page - 1) * pageSize)
            msg="current book review url is {} pageNum is {}".format(url,page)
            logger.info(msg)
            web_data = request_with_proxy(url, True)
            soup = BeautifulSoup(web_data.text, 'lxml')
            commments = soup.select('div.review-list > div > div.main.review-item > div.main-bd > h2 > a')
            temp=[]
            for comment in commments:
                temp.append(get_review_page(comment.get("href")))

            for item in temp:
                list.append(item)
                total_review_count=total_review_count+1
            if len(temp) == 0 or (page_limit is not None and page >= page_limit):
                break

            page = page + 1
            time.sleep(sleep_time)
        except requests.exceptions.ProxyError as e:
            print(e)
            print("代理异常.... 返回部分数据  " + "  当前页数为...", page)
            break
    dict={}
    dict['review_total_count']=total_review_count
    dict['reviews']=list
    return dict


# 获取书评页面的相关数据
def get_review_page(url, sleep_time=1):
    time.sleep(sleep_time)
    web_data = request_with_proxy(url, True)
    msg = "success catch book_reivew detail url is {}".format(url)
    logger.info(msg)
    soup = BeautifulSoup(web_data.text, 'lxml')
    title = soup.select("#wrapper > #content > div > div.article > h1 > span")
    content = soup.select("#link-report > div.review-content.clearfix")
    useful_count = soup.select("div.main-ft > div > div.main-panel-useful > button.btn.useful_count")
    useless_count = soup.select("div.main-ft > div > div.main-panel-useful > button.btn.useless_count")

    try:
        review_data = {
            'title': title[0].get_text(),
            'content': content[0].get_text(),
            'useful_count': int(get_count(useful_count[0].get_text())),
            'useless_count': int(get_count(useless_count[0].get_text()))
        }
    except IndexError as e:
        msg='当前书评页面有误  url is {} \n error is {}'.format(url,e)
        logger.info(msg)
        review_data = {
            'title': '',
            'content': '',
            'useful_count': 0,
            'useless_count': 0
        }

    return review_data


# 利用正则提取字符串中的数字
def get_count(str):
    obj = re.match('([^\d]*)(\d+)', str)
    if obj:
        try:
            return obj.group(2)
        except IndexError as e:
            print(str + '  except: 字符串中匹配不到数字　字符串有误')
            return -1
    else:
        print(str + "  不是我们正常需要的字符串　正确例子： 有用 585")


def get_book_data_from_api_byID(book_id,filter_field=None):
    if filter_field is None:
        url="https://api.douban.com/v2/book/"+book_id
    else:
        url = "https://api.douban.com/v2/book/" + book_id + "?fields=" + ','.join(filter_field)

    print(url)
    web_data = request_with_proxy(url, True)
    book_json=web_data.text
    book_dict=json.loads(book_json)
    try:
        book_dict['comments']=get_book_comments(book_id, page=1,sleep_time=1,page_limit=1)
        book_dict['reviews']=get_book_reviews(book_id, page=1,sleep_time=1,page_limit=1)

        msg="insert a book data into mongodb book_id is {}".format(book_id)
        logger.info(msg)
        print(book_dict)
        #mongo._insert(book_json)
    except IndexError as e:
        msg='当前书籍详细页面有误  url is {} \n\n error is {} \n\n response_data :\n\n  {}'.format(url,e,web_data.text)
        logger.info(msg)

# 根据书籍ID去获取一本书的详细数据
def get_book_data_byID(book_id):
    url = "https://book.douban.com/subject/" + book_id + "/"
    web_data = request_with_proxy(url, True)
    soup = BeautifulSoup(web_data.text, 'lxml')
    tags = soup.select("#db-tags-section > div.indent")
    name = soup.select("#wrapper > h1 > span")
    rate = soup.select("#interest_sectl > div > div.rating_self.clearfix > strong")
    rate_people = soup.select(
        "#interest_sectl > div > div.rating_self.clearfix > div > div.rating_sum > span > a > span")
    review_total_count = soup.select(
        "#content > div > div.article > div.related_info > section > header > h2 > span > a")
    comment_total_count = soup.select("#content > div > div.article > div.related_info > div.mod-hd > h2 > span.pl > a")
    info=soup.select("#info ")
    # content_desc=soup.select("#link-report > div > div.intro")
    # author_desc=soup.select("#content > div > div.article > div.related_info > div.indent > div > div.intro")
    try:
        print(1)
        print(name[0].get_text())
        print(2)
        print(float(rate[0].get_text()))
        print(3)
        print(int(rate_people[0].get_text()))
        print(4)
        print(list(tags[0].stripped_strings))
        print(5)
        print(int(get_count(review_total_count[0].get_text())))
        print(6)
        print(int(get_count(comment_total_count[0].get_text())))
        print(7)
        print(deal_info(info[0].get_text()))
        print(8)
        # print(content_desc[0].get_text())
        # print(9)
        # print(author_desc[0].get_text())
        # print(10)
        book = {
            'book_id': book_id,
            'name': name[0].get_text(),
            'rate': float(rate[0].get_text()),
            'rate_people': int(rate_people[0].get_text()),
            'tags': list(tags[0].stripped_strings),
            'review_total_count': int(get_count(review_total_count[0].get_text())),
            'comment_total_count': int(get_count(comment_total_count[0].get_text())),
            'book_info':deal_info(info[0].get_text()),
            'comments': get_book_comments(book_id,sleep_time=2),
            'reviews': get_book_reviews(book_id,sleep_time=2)
        }

        msg="insert a book data into mongodb book_id is {}".format(book_id)
        logger.info(msg)
        print(book)
        mongo._insert(book)
    except IndexError as e:
        msg='当前书籍详细页面有误  url is {} \n\n error is {} \n\n response_data :\n\n  {}'.format(url,e,web_data.text)
        logger.info(msg)



def deal_info(info_str):
    dict_info={}
    temp=""
    flag=0
    try:
        for line in info_str.split("\n"):
            line=line.replace(' ','').replace('\xa0','')
            if line == '':
                continue
            if line.find(":") != -1:
                if flag == 0:
                    temp+=line
                    flag=1
                    continue
                arr=temp.split(':')
                dict_info[arr[0]]=arr[1]
                temp = ""
            temp += line
        arr = temp.split(':')
        dict_info[arr[0]] = arr[1]
    except Exception as e:
        logger.info(e)
        logger.info("处理书籍基础信息时发生异常....")

    return dict_info



# json_string=json.dumps(_list, ensure_ascii=False, indent=4)#ensure_ascii=False 输出中文 indent 排版json格式










def changeip():
    global ip_port
    ip_port=ips.getRandomOne()
    msg="after {} seconds change ip in case ip will be blocked new ip is {}".format(changeIP_seconds,ip_port)
    logger.info(msg)


def request_with_proxy(url, request_proxy):
    global ip_port
    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "User-Agent": random.choice(PC_USER_AGENTS)
    }

    # 获取进入while循环的初始时间
    start_time=time.time()
    while True:

        #获取当前时间和之前的初始时间做比较，如果超出自定义的时间则raise requests.exceptions.ProxyError
        end_time=time.time()
        if int(end_time-start_time)>proxy_timeout:
            msg="request with proxy 方法时间执行过长 可能原因： IP池内IP全部失效或其他异常错误  当前ip为 {}".format(ip_port)
            raise requests.exceptions.ProxyError(msg)
        proxy = {
            'http': ip_port,
            'https': ip_port
        }

        if request_proxy:
            try:
                response = requests.get(url, proxies=proxy, timeout=request_timeout, headers=headers)
                code=response.status_code
                msg = "doing http request successfully current proxy ip is {} status_code :{}".format(ip_port,code)
                logger.info(msg)

                if code == 404:
                    msg=" 404 Client Error: Not Found for url:{}".format(url)
                    logger.info(msg)
                    return response

                response.raise_for_status()
                if code == 200 and custom_filter_str in response.text:
                    raise Exception

                return response
            except requests.HTTPError as e:
                logger.info(e)
                ip_port = ips.getRandomOne()
                msg = "random pick a ip from ipList new ip is {}".format(ip_port)
                logger.info(msg)
            except Exception as e:
                msg ="ip is {} can't use ".format(ip_port)
                logger.info(msg)
                ip_port = ips.getRandomOne()
                msg="random pick a ip from ipList new ip is {}".format(ip_port)
                logger.info(msg)
        else:
            try:
                response = requests.get(url, timeout=request_timeout, headers=headers)
                return response
            except Exception as e:
                msg = "ip is {} can't use ".format(ip_port)
                logger.info(msg)
                ip_port = ips.getRandomOne()
                msg = "random pick a ip from ipList new ip is {}".format(ip_port)
                logger.info(msg)


if __name__ == '__main__':

    # 第一次默认的IP
    ip_port = ips.get_one(1)

    # scheduler = BackgroundScheduler()
    # scheduler.add_job(changeip, 'interval', seconds=changeIP_seconds)
    # scheduler.start()
    #
    # 金融 管理 科技 励志 传记 小说 心理学 历史 爱情 养生
    get_books_data("金融",exclude_collection=['book_rumen','book_jinjie'],page=1,page_limit=5)


    #get_book_data_byID("19952400")
    # field=['rating','subtitle','author','pubdate','tags','origin_title','image','binding','translator','pages','images','id','publisher','alt','isbn10','isbn13','title','url','alt_title','author_intro','summary','price']
    # get_book_data_from_api_byID("19952400",field)

    #json_data=request_with_proxy("https://book.douban.com/review/8774415/",True)
    #print(json_data.text)

    # list = get_book_reviews("19952400")
    # json_string = json.dumps(list, ensure_ascii=False, indent=4)
    # print(json_string)

