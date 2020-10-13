# -*- coding: utf-8 -*-
# @Time    : 2020/4/7 14:50
# @Author  : Aurora
# @File    : create_wholeset.py
# @Function: create wholeset_ratings.txt and wholeset_movies.txt

path_mid2nid = '../raw_data/mid2nid.txt'
path_nid2mid = '../raw_data/nid2mid.txt'
path_ratings = '../raw_data/ratings.txt'
path_movies = '../raw_data/movies.txt'
path_wholeset_ratings = '../data/wholeset_ratings.txt'
path_wholeset_movies = '../data/wholeset_movies.txt'

scores = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
genres_dict = {
    'None': '0',
    'Action': '1',
    'Adventure': '2',
    'Animation': '3',
    'Children': '4',
    'Comedy': '5',
    'Crime': '6',
    'Documentary': '7',
    'Drama': '8',
    'Fantasy': '9',
    'Film-Noir': '10',
    'Horror': '11',
    'Musical': '12',
    'Mystery': '13',
    'Romance': '14',
    'Sci-Fi': '15',
    'Thriller': '16',
    'War': '17',
    'Western': '18',
}

def loadDict(filename):
    """
    load dict from file
    :param filename: path of mid2nid.txt or nid2mid.txt
    :return: dict mid2nid: key=movieId, value=nodeId
          or dict nid2mid: key=nodeId, value=movieId
    """
    dict = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            key, value = line.strip().split(';')
            dict[key] = value
    return dict

def create_wholesetRatings(path_ratings, mid2nid):
    """
    create wholeset_ratings.txt,
    convert movieId in ratings.txt to nodeId
    :param path_ratings: path of ratings.txt
    :param mid2nid: dict mid2nid: key=movieId, value=nodeId
    :return:
    """
    ratings = open(path_ratings, 'r', encoding='utf-8')
    with open(path_wholeset_ratings, 'a+', encoding='utf-8') as f:
        for line in ratings:
            usrId, movieId, rating, _=line.strip().split(';')
            f.write(usrId + ';' + mid2nid[movieId] + ';' + rating+'\n')

def create_wholesetMovies(path_movies, mid2nid):
    """
    create wholeset_movies.txt,
    convert movieId in movies.txt to nodeId
    convert genres in movies.txt to genres code
    :param path_movies: path of movies.txt
    :param mid2nid: dict mid2nid: key=movieId, value=nodeId
    :return:
    """
    movies = open(path_movies, 'r', encoding='utf-8')
    with open(path_wholeset_movies, 'a+', encoding='utf-8') as f:
        for line in movies:
            # movieId, title, genres = line.strip().split(';')
            # titles of some movies involve ';'
            contents = line.strip().split(';')
            movieId = contents[0]
            genres = contents[-1]

            if movieId not in mid2nid.keys(): continue

            genre_encoded = ''
            for index, genre in enumerate(genres.split('|')):
                if genre not in genres_dict.keys():
                    continue
                else:
                    genre_encoded += genres_dict[genre] + ','
            genre_encoded = genre_encoded[:-1]
            f.write(mid2nid[movieId] + ';' + genre_encoded + '\n')


if __name__ == '__main__':
    mid2nid_dict = loadDict(path_mid2nid)
    nid2mid_dict = loadDict(path_nid2mid)
    # print(mid2nid_dict, '\n')
    # print(nid2mid_dict, '\n')
    # create_wholesetRatings(path_ratings, mid2nid_dict)
    # create_wholesetMovies(path_movies, mid2nid_dict)
    pass