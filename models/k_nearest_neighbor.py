from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

import numpy as np

class K_Nearest_Neighbor(object):
    def __init__(self):
        pass

    def train(self, matrix_train):
        self.similarity = cosine_similarity(X=matrix_train, Y=None, dense_output=True)
        print self.similarity
    def predict(self, matrix_train, k):
        prediction_scores = []
        for user_index in tqdm(range(matrix_train.shape[0])):
            # Get user u's prediction scores for all items
            vector_u = self.similarity[user_index]

            # Get closest K neighbors excluding user u self
            similar_users = vector_u.argsort()[::-1][1:k+1]

            # Get neighbors similarity weights and ratings
            similar_users_weights = self.similarity[user_index][similar_users]
            similar_users_ratings = matrix_train[similar_users].toarray()

            prediction_scores_u = similar_users_ratings * similar_users_weights[:, np.newaxis]

            prediction_scores.append(np.sum(prediction_scores_u, axis=0))

        return np.array(prediction_scores)
