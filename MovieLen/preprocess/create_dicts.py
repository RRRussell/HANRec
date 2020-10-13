# -*- coding: utf-8 -*-
# @Time    : 2020/4/7 13:57
# @Author  : Aurora
# @File    : create_dicts.py
# @Function: to create two dictionaries: movieId -> nodeId; nodeId -> movieId

import numpy as np


path_ratings_txt = '../raw_data/ratings.txt'
path_mid2nid_txt = '../raw_data/mid2nid.txt'
path_nid2mid_txt = '../raw_data/nid2mid.txt'
path_mvid2vid_txt = '../transfer_dicts/mvid2vid.txt'
path_usrid2uid_txt = '../transfer_dicts/usrid2uid.txt'
path_vid2mvid_txt = '../transfer_dicts/vid2mvid.txt'


def ratingsInfo():
    ratings = np.loadtxt(path_ratings_txt, dtype=float, encoding='utf-8',delimiter=';')
    print('ratings:', ratings)
    uni_userId = np.unique(ratings[:, 0].astype(np.int))
    uni_movieId = np.unique(ratings[:, 1].astype(np.int))
    uni_rating = np.unique(ratings[:, 2])
    # print('uni_userId\n', uni_userId)
    # print('uni_movieId\n', uni_movieId)
    # print('uni_rating\n', uni_rating)
    print('uni_userId shape', uni_userId.shape)
    print('uni_movieId shape', uni_movieId.shape)
    print('uni_rating shape', uni_rating.shape)

def create_mid2nid():
    ratings = np.loadtxt(path_ratings_txt, dtype=int, encoding='utf-8',delimiter=';')
    uni_userId = np.unique(ratings[:, 0])
    uni_movieId = np.unique(ratings[:, 1])
    dict = {}
    for index, item in enumerate(uni_movieId):
        dict[item] = index+len(uni_userId)+1
    with open(path_mid2nid_txt, 'a+', encoding='utf-8') as f:
        for key in dict.keys():
            f.write((str(key)+';'+str(dict[key])+'\n'))

def create_nid2mid():
    ratings = np.loadtxt(path_ratings_txt, dtype=int, encoding='utf-8',delimiter=';')
    uni_userId = np.unique(ratings[:, 0])
    uni_movieId = np.unique(ratings[:, 1])
    n2m_dict = {}
    for index, mid in enumerate(uni_movieId):
        n2m_dict[index+len(uni_userId)+1] = mid
    with open(path_nid2mid_txt, 'a+', encoding='utf-8') as f:
        for key in n2m_dict.keys():
            f.write((str(key)+';'+str(n2m_dict[key])+'\n'))

def create_rawId2encodedId():
    ratings = np.loadtxt(path_ratings_txt, dtype=float, encoding='utf-8',delimiter=';')
    uni_userId = np.unique(ratings[:, 0].astype(np.int))
    uni_movieId = np.unique(ratings[:, 1].astype(np.int))
    uni_rating = np.unique(ratings[:, 2])
    usrid2uid_dict = {}
    mvid2vid_dict = {}
    for index, mid in enumerate(uni_movieId):
        mvid2vid_dict[mid] = index
    for index, uid in enumerate(uni_userId):
        usrid2uid_dict[uid] = index
    print('user id to uid\n', usrid2uid_dict)
    print('movie id to vid\n', mvid2vid_dict)
    with open(path_mvid2vid_txt, 'a+', encoding='utf-8') as f:
        for key in mvid2vid_dict.keys():
            f.write(str(key)+';'+str(mvid2vid_dict[key])+'\n')

    with open(path_usrid2uid_txt, 'a+', encoding='utf-8') as f:
        for key in usrid2uid_dict.keys():
            f.write(str(key)+';'+str(usrid2uid_dict[key])+'\n')

def create_encodedId2rawId():
    pass

if __name__ == '__main__':
    # create_mid2nid()
    # create_nid2mid()
    ratingsInfo()
    # create_rawId2encodedId()