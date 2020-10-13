# -*- coding: utf-8 -*-
# @Time    : 2020/4/15 17:53
# @Author  : Aurora
# @File    : create_wholeset_new.py
# @Function: 

path_ratings = '../raw_data/ratings.txt'
path_movies = '../raw_data/movies.txt'

path_mvid2vid_txt = '../transfer_dicts/mvid2vid.txt'
path_usrid2uid_txt = '../transfer_dicts/usrid2uid.txt'

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

def create_wholesetRatings(path_ratings, usrid2uid, mvid2vid):
    ratings = open(path_ratings, 'r', encoding='utf-8')
    with open(path_wholeset_ratings, 'a+', encoding='utf-8') as f:
        for line in ratings:
            usrId, movieId, rating, _=line.strip().split(';')
            rating_f = float(rating)
            f.write(usrid2uid[usrId] + ';' + mvid2vid[movieId] + ';' + str(int(rating_f*2-1))+'\n')

def create_wholesetMovies(path_movies, mvid2vid):
    movies = open(path_movies, 'r', encoding='utf-8')
    with open(path_wholeset_movies, 'a+', encoding='utf-8') as f:
        for line in movies:
            # movieId, title, genres = line.strip().split(';')
            # titles of some movies involve ';'
            contents = line.strip().split(';')
            movieId = contents[0]
            genres = contents[-1]

            if movieId not in mvid2vid.keys(): continue

            genre_encoded = ''
            for index, genre in enumerate(genres.split('|')):
                if genre not in genres_dict.keys():
                    continue
                else:
                    genre_encoded += genres_dict[genre] + ','
            genre_encoded = genre_encoded[:-1]
            f.write(mvid2vid[movieId] + ';' + genre_encoded + '\n')

if __name__ == '__main__':
    usrid2uid_dict = loadDict(path_usrid2uid_txt)
    mvid2vid_dict = loadDict(path_mvid2vid_txt)
    # print('user id to uid:\n', usrid2uid_dict)
    # print('movie id to vid:\n', mvid2vid_dict)
    # create_wholesetRatings(path_ratings, usrid2uid_dict, mvid2vid_dict)
    # create_wholesetMovies(path_movies, mvid2vid_dict)