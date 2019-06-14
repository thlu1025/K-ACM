# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 16:54:22 2019

@author: lth
"""
import os

def getParent():
    return os.path.abspath(os.path.join(os.getcwd(), ".."))

def createDir(dirName):
    parent_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
    if not os.path.exists(parent_path + dirName):
        os.makedirs(parent_path + dirName)
