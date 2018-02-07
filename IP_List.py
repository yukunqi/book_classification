#!/usr/bin/env python
# -*- coding: utf-8 -*-


from spider.data_save import pipeline
from proxy_basic_config import collection_name
from apscheduler.schedulers.background import BackgroundScheduler
from AiSpider.spider.log_format import spider_log
import time
import random

logger=spider_log(log_name="Ips")

class IPs(object):
    def __init__(self,collection_name):
        self.collection_name=collection_name
        self.ipList=[]
        self.getIPList()

    def getIPList(self):
        self.ipList=pipeline.get_all_IP(self.collection_name)
        msg="get ip list from {}".format(collection_name)
        logger.info(msg)

    def getSize(self):
        if self.ipList is None:
            raise Exception

        return len(self.ipList)

    def get_one(self,index):
        len=self.getSize()

        if self.ipList is None or len == 0:
            raise IndexError

        if index >= len:
            index = 0

        return self.ipList[index]

    def getRandomOne(self):
        return random.choice(self.ipList)

if __name__ == '__main__':
    ips=IPs(collection_name)
    scheduler=BackgroundScheduler()
    scheduler.add_job(ips.getIPList,'interval',seconds=10)
    scheduler.start()
    try:
        # This is here to simulate application activity (which keeps the main thread alive).
        while True:
            time.sleep(2)
            ip=ips.get_one(1)
            print(ip)
    except (KeyboardInterrupt, SystemExit):
        # Not strictly necessary if daemonic mode is enabled but should be done if possible
        scheduler.shutdown()