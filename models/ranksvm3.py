from prediction.predictor import predict_scores, predict_vector
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
from utils.critique import lpranksvm3
from utils.keyphrase_selection import *
import copy
from tqdm import tqdm
import numpy as np
import time
from utils.progress import inhour

class RankSVM3(object):
    def __init__(self, keyphrase_freq, item_keyphrase_freq, row, matrix_Train, matrix_Test, test_users,
                 target_ranks, num_items_sampled, num_keyphrases, df,
                 max_iteration_threshold, keyphrase_popularity, dataset_name,
                 model, parameters_row, keyphrases_names, keyphrase_selection_method, max_wanted_keyphrase, lamb, **unused):
        self.keyphrase_freq = keyphrase_freq
        self.item_keyphrase_freq = item_keyphrase_freq
        self.row = row
        self.matrix_Train = matrix_Train
        self.num_users, self.num_items = matrix_Train.shape
        self.matrix_Test = matrix_Test
        self.test_users = test_users
        self.target_ranks = target_ranks
        self.num_items_sampled = num_items_sampled
        self.num_keyphrases = num_keyphrases
        self.df = df
        self.max_iteration_threshold = max_iteration_threshold
        self.keyphrase_popularity = keyphrase_popularity
        self.dataset_name = dataset_name
        self.model = model
        self.parameters_row = parameters_row
        self.keyphrase_selection_method = keyphrase_selection_method
        self.max_wanted_keyphrase = max_wanted_keyphrase
        
        self.lamb = lamb
        self.keyphrases_names = keyphrases_names

    def start_critiquing(self):
        self.get_initial_predictions() 

        for user in tqdm(self.test_users):
            start_time = time.time()
            
            # User id starts from 0
            self.row['user_id'] = user

            initial_prediction_items = predict_vector(rating_vector=self.prediction_scores[user],
                                                            train_vector=self.matrix_Train[user],
                                                            remove_train=True)
            # For keyphrase selection method 'diff' 
            top_recommended_keyphrase_freq = get_item_keyphrase_freq(self.item_keyphrase_freq,item = initial_prediction_items[0])
            
            # The iteration will stop if the wanted item is in top n
            for target_rank in self.target_ranks:
                self.row['target_rank'] = target_rank
                
                # Pick wanted items in test items
                candidate_items = self.matrix_Test[user].nonzero()[1]
                train_items = self.matrix_Train[user].nonzero()[1]
                wanted_items = np.setdiff1d(candidate_items, train_items)
                
                for item in wanted_items:
                    # Item id starts from 0
                    self.row['item_id'] = item
                    
                    ## Get item name
                    # try:
                    #     self.row['item_name'] = get_restaurant_name(df_train, self.business_df,item)
                    # except:
                    #     self.row['item_name'] = 'NOT_FOUND'
                    
                    # Set the wanted item's initial rank as None
                    self.row['item_rank'] = None
                    # Set the wanted item's initial prediction score as None
                    self.row['item_score'] = None
                    
                    if self.keyphrase_selection_method == "random" or self.keyphrase_selection_method == "pop":
                        # Get the item's existing keyphrases (we can boost)
                        try:
                            remaining_keyphrases = self.item_keyphrase_freq[item].nonzero()[1]
                        except:
                            remaining_keyphrases = np.ravel(self.item_keyphrase_freq[item].nonzero())
                    if self.keyphrase_selection_method == "diff":
                        # For keyphrase selection method 'diff' 
                        target_keyphrase_freq = get_item_keyphrase_freq(self.item_keyphrase_freq,item = item)
                        diff_keyphrase_freq = target_keyphrase_freq - top_recommended_keyphrase_freq
                        remaining_keyphrases = np.argsort(np.ravel(diff_keyphrase_freq))[::-1][:self.max_wanted_keyphrase]
                        
                    self.row['num_existing_keyphrases'] = len(remaining_keyphrases)
                    
                    if len(remaining_keyphrases) == 0:
                        break
                    
                    self.row['iteration'] = 0
                    self.row['critiqued_keyphrase'] = None
                    self.row['result'] = None
                    self.df = self.df.append(self.row, ignore_index=True)

                    query = []
                    affected_items = np.array([])
                    
                    # Set up latent embedding
                    user_latent_embedding = [self.Y[user]]
                    
                    for iteration in range(self.max_iteration_threshold):
                        self.row['iteration'] = iteration + 1            
                        
                        if self.keyphrase_selection_method == "pop":
                            # Always critique the least popular keyphrase
                            critiqued_keyphrase = remaining_keyphrases[np.argmin(self.keyphrase_popularity[remaining_keyphrases])]
                            
                        elif self.keyphrase_selection_method == "random":
                            critiqued_keyphrase = np.random.choice(remaining_keyphrases, size=1, replace=False)[0]
            
                        elif self.keyphrase_selection_method == "diff":
                            critiqued_keyphrase = remaining_keyphrases[0]
                        
                        self.row['critiqued_keyphrase'] = critiqued_keyphrase
                        self.row['critiqued_keyphrase_name'] = self.keyphrases_names[critiqued_keyphrase]
                        query.append(critiqued_keyphrase)

                        # Get affected items (items have critiqued keyphrase)
                        current_affected_items = self.item_keyphrase_freq[:, critiqued_keyphrase].nonzero()[0]
                        affected_items = np.unique(np.concatenate((affected_items, current_affected_items))).astype(int)
                        unaffected_items = np.setdiff1d(range(self.num_items), affected_items)

                        if iteration == 0:
                            prediction_items = initial_prediction_items #calculated once for each user

                        affected_items_mask = np.in1d(prediction_items, affected_items)
                        affected_items_index_rank = np.where(affected_items_mask == True)
                        unaffected_items_index_rank = np.where(affected_items_mask == False)

                        ## concat critique embeddings to user latent embedding
                        # Get critique vector 
                        critiqued_vector = np.zeros(self.keyphrase_freq.shape[1])
                        critiqued_vector[critiqued_keyphrase] = max(self.keyphrase_freq[user , critiqued_keyphrase],1)
                        
                        # map user critique to user latent embedding
                        k_ci = self.reg.predict(critiqued_vector.reshape(1, -1)).flatten()
                        user_latent_embedding.append(k_ci)
                        
                        prediction_scores_u, thetas = lpranksvm3(initial_prediction_u=self.prediction_scores[user],
                                                                             keyphrase_freq=copy.deepcopy(self.keyphrase_freq),
                                                                             affected_items=np.intersect1d(affected_items, prediction_items[affected_items_index_rank[0][:100]]),
                                                                             unaffected_items=np.intersect1d(unaffected_items, prediction_items[unaffected_items_index_rank[0][:100]]),
                                                                             num_keyphrases=self.num_keyphrases,
                                                                             query=query,
                                                                             test_user=user,
                                                                             item_latent=self.RQ,
                                                                             reg=self.reg,
                                                                             user_latent_embedding = user_latent_embedding,
                                                                             item_keyphrase_freq = self.item_keyphrase_freq,
                                                                             Y = self.Y,
                                                                             lamb = self.lamb)
                        self.row['theta'] = thetas
                        prediction_items = predict_vector(rating_vector=prediction_scores_u,
                                                          train_vector=self.matrix_Train[user],
                                                          remove_train=False)
                        recommended_items = prediction_items
                        
                        # Current item rank
                        item_rank = np.where(recommended_items == item)[0][0]

                        self.row['item_rank'] = item_rank
                        self.row['item_score'] = prediction_scores_u[item]

                        if item_rank + 1 <= target_rank:
                            # Items is ranked within target rank
                            self.row['result'] = 'successful'
                            self.df = self.df.append(self.row, ignore_index=True)
                            break
                        else:
                            remaining_keyphrases = np.setdiff1d(remaining_keyphrases, critiqued_keyphrase)
                            # Continue if more keyphrases and iterations remained
                            if len(remaining_keyphrases) > 0 and self.row['iteration'] < self.max_iteration_threshold:
                                self.row['result'] = None
                                self.df = self.df.append(self.row, ignore_index=True)
                            else:
                                # Otherwise, mark fail
                                self.row['result'] = 'fail'
                                self.df = self.df.append(self.row, ignore_index=True)
                                break
        
            print("User", user ,"Elapsed: {}".format(inhour(time.time() - start_time)))
        return self.df


    def get_initial_predictions(self):
        # self.business_df = get_business_df()
        self.Y, RQt , Bias = self.model(self.matrix_Train,
                                       iteration=self.parameters_row['iter'],
                                       lamb=self.parameters_row['lambda'],
                                       rank=self.parameters_row['rank'])
        
        self.RQ = RQt.T
        self.reg = LinearRegression().fit(normalize(self.keyphrase_freq), self.Y)

        self.prediction_scores = predict_scores(matrix_U=self.RQ,
                                                matrix_V=self.Y,
                                                bias=Bias).T

