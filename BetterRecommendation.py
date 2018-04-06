import pandas as pd
import numpy as np
import operator
from mapreduce import mapreduce
from IPython.display import display, HTML
from numpy import dot
from numpy.linalg import norm


def combinations(iterable, r):
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
        for j in range(i + 1, r):
            indices[j] = indices[j - 1] + 1
        yield tuple(pool[i] for i in indices)


def main():
    ratings = read_netflix_ratings()
    map_reducer = mapreduce()
    pipeline = map_reducer.parallelize(ratings, 128)

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
    return df



def read_netflix_ratings():
    df = pd.read_csv('data/ratings.csv')
    ratings = df.values.tolist()
    return ratings


def reducer(a, b):
    return a + b


def mapper1(record):
    result = (int(record[1]), [(int(record[0]), record[2])])
    #     print('mapper1: %s-> %s'%(record,result))
    return result


def mapper2(record):
    result = []
    number = len(record[1])
    if number >= 200:
        result = [(element[0], [(record[0], element[1])]) for element in record[1]]
    # print('mapper2: %s-> %s'%(record,result))
    return result


def mapper3(record):
    """
    return: [( ( movie_id1, movie_id2 ), [(rating, rating)]), ...]
    """
    # YOUR CODE HERE
    movie_rate = record[1]
    comb = list(combinations(movie_rate, 2))
    result = []
    for i in comb:
        result.append(((i[0][0], i[1][0]), [(i[0][1], i[1][1])]))
    # print('mapper3: %s -> %s'%(record,result))
    return result


def mapper4(record):
    """
    calculate similarity(cosine distance) of two rating vectors for [(movie1,movie2)]
    return [(movie1,[movie2,similarity])]

    """
    rate = record[1]
    rate1 = []
    rate2 = []
    for i in range(len(rate)):
        rate1.append(rate[i][0])
        rate2.append(rate[i][1])
    cos_sim = dot(rate1, rate2) / (norm(rate1) * norm(rate2))
    result = [(record[0][0], [(record[0][1], cos_sim)])]
    #     print('mapper4: %s -> %s'%(record,result))
    return result


def mapper5(record):
    result = []
    if len(record[1]) <= 3:
        result.append(record)
    else:
        sort_list = sorted(record[1], key=operator.itemgetter(1), reverse=True)
        result.append((record[0], sort_list[:3]))
    # print('mapper5: %s-> %s'%(record,result))
    return result


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