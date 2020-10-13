# -*- coding: utf-8 -*-
# @Time    : 2020/4/7 13:53
# @Author  : Aurora
# @File    : convert_csv2txt.py
# @Function: convert movies.csv, ratings.csv to txt format. content is seperated by ';'

import pandas as pd


path_ratings = '../raw_data/ratings.csv'
path_ratings_txt = '../raw_data/ratings.txt'
path_movies = '../raw_data/movies.csv'
path_movies_txt = '../raw_data/movies.txt'

def convertRatings():
    ratings = pd.read_csv(path_ratings, encoding='utf-8')
    with open(path_ratings_txt, 'a+', encoding='utf-8') as f:
        for line in ratings.values:
            f.write((str(int(line[0])) + ';' + str(int(line[1])) + ';' + str((line[2])) + ';' + str(
                int(line[3])) + '\n'))

def convertMovies():
    movies = pd.read_csv(path_movies, encoding='utf-8')
    with open(path_movies_txt, 'a+', encoding='utf-8') as f:
        for line in movies.values:
            title=''
            for item in line[1:-1]:
                title += str(item)
            f.write((str(int(line[0])) + ';' + title + ';' + str(line[-1]) + '\n'))

if __name__ == '__main__':
    # convertMovies()
    # convertRatings()
    pass