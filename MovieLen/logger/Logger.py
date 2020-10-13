# -*- coding: utf-8 -*-
# @Time    : 2020/4/18 16:11
# @Author  : Aurora
# @File    : Logger.py
# @Function: 

import sys

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass