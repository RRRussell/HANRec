############################################################################
# README for preprocessed MovieLens by weihao
# Original Datasets:
# MovieLens Latest Datasets https://grouplens.org/datasets/movielens/latest/
############################################################################

wholeset_ratings.txt
===================
format		nodeId(user);nodeId(movie);rating
example		590;2380;3

uId:	    from 0 to 609
vId:	    from 0 to 9723
rating:		from 0 to 9 (0 represent 0.5 in 5-star scale, 9 represents 5.0 in 5-star scale)


wholeset_movies.txt
===================
format		nodeId(movie);encoded genres
example		9713;1,2,5,15

vID:	    from 0 to 9723
encoded genres:	encoded by dict
{
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
multiple encoded genres are separated by ','