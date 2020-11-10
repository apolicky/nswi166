#!/usr/bin/env python3

import time
import random
import pandas as pd
from scipy.sparse import csr_matrix as sparse_matrix
from sklearn.neighbors import NearestNeighbors as nn


K = 30


def get_ratings():
    start = time.time()
    with open('ratings.dat') as ratings_file:
        df_r = pd.read_csv(ratings_file, delimiter="::", usecols=['userId', 'movieId', 'rating'], dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'}, engine="python")

    with open('movies.dat') as movies_file:
        df_m = pd.read_csv(movies_file, delimiter="::", usecols=['movieId', 'title'], dtype={'movieId': 'int32', 'title': 'str'}, engine="python")

    data = df_r.pivot(index='movieId',columns='userId',values='rating').fillna(0)

    ''' title_id: movieTitle -> movieId'''
    title_id = {
        movie: i for i, movie in
        enumerate(list(df_m.set_index('movieId').loc[data.index].title))  # noqa
    }

    ''' id_title: movieId -> movieTitle'''
    id_title = {
        i: movie for i, movie in
        enumerate(list(df_m.set_index('movieId').loc[data.index].title))  # noqa
    }

    end = time.time()
    print("data loaded in {}".format(end - start))

    # return data, data_train, data_test, title_id, id_title
    return data, title_id, id_title


# def find_matches(model, data, target, nr_recommends, id_movie):
def find_matches(model, data, target, nr_recommends, id_movie=None):
    distances, indices = model.kneighbors(data[target], nr_recommends+1)

    a = sorted(list(zip(distances.squeeze().tolist()[1:], indices.squeeze().tolist()[1:])), reverse=True)

    if id_movie is None:
        return a
    else:
        print('target_id {}, name: {}'.format(target,id_movie[target]))
        for dist, m_id in a:
            print('{}: {}'.format(id_movie[m_id], dist))


def train_model(data, k=K):
    start = time.time()
    model = nn()
    model.set_params(n_neighbors=k, algorithm='auto', metric='cosine')
    model.fit(data)
    end = time.time()
    print("training took {}".format(end-start))
    return model


"""for randomly chosen users hides some of their ratings"""
def hide_data(data):
    users = random.sample(range(data.shape[1]), k=int(round(0.1 * data.shape[1])))
    user_hidden = dict()
    for u in users:
        rated_movies = list(data.getcol(u).nonzero()[0])
        user_hidden[u] = random.sample(rated_movies, k=int(round(0.2 * len(rated_movies))))
        data[user_hidden[u], u] = 0

    return users, user_hidden


def evaluate_model(model, data, users, user_hidden):
    hits = misses = users_scanned = 0
    recommendation_times = []

    for u in users:
        start = time.time()
        u_ratings = data.getcol(u).nonzero()[0]
        u_avg_rating = sum(data[u_ratings, u].data) / len(data[u_ratings, u].data)
        u_movies = list(
            filter(lambda x: data[x, u] >= u_avg_rating, data.getcol(u).nonzero()[0]))

        m_dist = {}
        for m in u_movies:
            for dist, other_movie in find_matches(model, data, m, K):
                try:
                    m_dist[other_movie] += dist
                except KeyError:
                    m_dist[other_movie] = dist

        m_dist_sorted = {k: v for k, v in sorted(m_dist.items(), key=lambda item: item[1], reverse=True)}

        topK = list(m_dist_sorted.keys())[:K]

        for recommendation in topK:
            if recommendation in user_hidden[u]:
                hits += 1
            else:
                misses += 1

        recommendation_times.append(time.time() - start)

        users_scanned += 1
        if users_scanned % 100 == 99:
            break

    print("hits: {}, missed:{}, hit rate:{}".format(hits, misses, hits / misses))
    print("avg recommendation time {}".format(sum(recommendation_times) / len(recommendation_times)))
    print(recommendation_times)


#########################################


data, title_id, id_title = get_ratings()

data = sparse_matrix(data)

users, user_hidden = hide_data(data)

model = train_model(data, k=K)

evaluate_model(model, data, users, user_hidden)






