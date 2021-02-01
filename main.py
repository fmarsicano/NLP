from movie_reviews import movie_review
from new_york_times import new_york_times
from utils import preprocess

if __name__ == '__main__':
    point = 1
    if point == 1:
        movie_review()
    else:
        new_york_times()
