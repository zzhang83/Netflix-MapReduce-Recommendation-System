import pandas as pd
import numpy as np
from pprint import pprint
import operator
from mapreduce import mapreduce
from pymongo import *
from IPython.display import display, HTML

############# util ###########################################################
def jaccard_similarity(n_common, n1, n2):
    # http://en.wikipedia.org/wiki/Jaccard_index
    numerator = n_common
    denominator = n1 + n2 - n_common
    if denominator == 0:
        return 0.0
    return numerator / denominator


def combinations(iterable, r):
    # http://docs.python.org/2/library/itertools.html#itertools.combinations
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(list(range(r))):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)
######## mapreduce function ####################################################


def main():
    # Get the ratings from ratings.csv
    ratings = read_netflix_ratings()

    # Initialize MapReduce
    map_reducer = mapreduce()
    pipeline = map_reducer.parallelize(ratings, 128)

    # DO NOT MODIFY THIS!
    similar_table = pipeline.map(mapper1) \
        .reduceByKey(reducer) \
        .flatMap(mapper2) \
        .reduceByKey(reducer) \
        .flatMap(mapper3) \
        .reduceByKey(reducer) \
        .flatMap(mapper4) \
        .reduceByKey(reducer) \
        .flatMap(mapper5)
    recommend_result = []
    print('******************************* Recommendation results ***********************************')
    for item in similar_table.collect():
        recommend_result.append(item)
        print(item)
    print('*********************************** Task 6 *****************************************')
    df = task_6(recommend_result)
    df2 = task_6_2(recommend_result)
    display(df2)
    return df


"""
TASK 0: Read in the netflix ratings from 'ratings.csv'
Input: None
Output: Two dimensional array where each record has a userId,movieId,rating,timestamp
"""
def read_netflix_ratings():
    # YOUR CODE HERE
    df = pd.read_csv('data/ratings.csv')
    ratings = df.values.tolist()
    return ratings

def read_netflix_movies():
    df = pd.read_csv('data/movies.csv')
    return df

def reducer(a, b):
    return a + b


def mapper1(record):
    """
    :param record: "user_id, movie_id, rating, timestamp"
    :return: (key, value)
              key: movie_id
              value: [(user_id, rating)]
    """
    # YOUR CODE HERE
    result = (int(record[1]), [(int(record[0]), record[2])])
    # print('mapper1: %s-> %s'%(record,result))
    return result


def mapper2(record):
    """
    :param record: (movie_id, [(user_id, rating), ...])
    :return: [(user_id, [(movie_id, rating, num_rater)]), ...]
    """
    # YOUR CODE HERE
    result = []
    number = len(record[1])
    if number >= 200:                           # will be useful in mapper3，  only keep movies with at least 200 ratings
        result = [(element[0], [(record[0], element[1], number)]) for element in record[1]]
    #print('mapper2: %s-> %s'%(record,result))
    return result


def mapper3(record):
    """
    to generate tuple that rated by the same user
    :param record: (user_id, [(movie_id, rating, num_rater), ...])
    :return: [( ( movie_id1, movie_id2 ), [(num_rater1, num_rater2)] ), ...]
    """
    # YOUR CODE HERE
    movie_rate = record[1]
    comb = list(combinations(movie_rate, 2))   # get pairs using combinations
    result = []
    for i in comb:
        result.append(((i[0][0], i[1][0]), [(i[0][2], i[1][2])]))    # only get moive pair and num_rater pair, no rating any more
    # print('mapper3: %s -> %s'%(record,result))
    return result


def mapper4(record):
    """
    :param record: (key, value)
                     key: (movie_id1, movie_id2)
                     value: [(num_rater1, num_rater2), ...]
    :return: [(key, value)] or []
               key: movie_id
               value: [(movie_id2, jaccard)]
    """
    # YOUR CODE HERE
    n_common = len(record[1])          # watch both movie1 and movie2
    n1 = record[1][0][0]               # number of users who watch movie1
    n2 = record[1][0][1]               # number of users who watch movie2
    if n_common >= 3:                  # Only  output  records  when  there  are  at  least  three pairs
        jaccard = jaccard_similarity(n_common, n1, n2)
        result = [(record[0][0], [(record[0][1], jaccard)])]
    # print('mapper4: %s -> %s'%(record,result))
    return result


def mapper5(record):
    """
    :param record: (key, value)
                    key: movie_id1
                    value: [(movie_id2, jaccard), ...]
    :return: (key, value)
              key: movie_id1
              value: top n item
    """
    # YOUR CODE HERE
    result = []
    if len(record[1])<=3:
        result.append(record)
    else:
        sort_list = sorted(record[1],key = operator.itemgetter(1),reverse = True)
        result.append((record[0],sort_list[:3]))
    # print('mapper5: %s-> %s'%(record,result))
    return result


def task_6_2(result):
    movies = df = pd.read_csv('data/movies.csv')
    movie_id_table = []
    for element in result:
        a = [element[0]]
        for i in range(len(element[1])):
            a.append(element[1][i][0])
        movie_id_table.append(a)

    title_table = []
    for element in movie_id_table:
        a = []
        for i in range(len(element)):
            a.append(movies['title'][element[i] - 1])
        title_table.append(a)
    title_df = pd.DataFrame(title_table, columns=['Movie', 'Moive you might like 1', 'Movie you might like 2',
                                                  'Movie you might like 3'])
    return title_df


def task_6(record):
    movies = df = pd.read_csv('data/movies.csv')
    movie_id_table = []
    for element in record:
        a = [element[0]]
        for i in range(len(element[1])):
            a.append(element[1][i][0])
        movie_id_table.append(a)

    title_genre = []
    for element in movie_id_table:
        a = []
        for i in range(len(element)):
            a.append([movies['title'][element[i] - 1], movies['genres'][element[i] - 1]])
        title_genre.append(a)
    title_genre = pd.DataFrame(title_genre, columns=['Movie', 'Moive you might like 1', 'Movie you might like 2',
                                                     'Movie you might like 3'])
    return title_genre

# Calls main() when you run the program
# if __name__ == '__main__':
#    main()


