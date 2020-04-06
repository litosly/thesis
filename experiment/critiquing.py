from utils.critique import sample_users
from utils.modelname import critiquing_models

import numpy as np
import pandas as pd


def critiquing(matrix_Train, matrix_Test, keyphrase_freq, dataset_name, model,
               parameters_row, critiquing_model_name, lamb, keyphrases_names, keyphrase_selection_method, item_keyphrase_freq=None, num_users_sampled=10,
               num_items_sampled=5, max_iteration_threshold=20):

    num_users = matrix_Train.shape[0]
    num_keyphrases = keyphrase_freq.shape[1]

    keyphrase_popularity = np.sum(item_keyphrase_freq, axis=1)

    columns = ['user_id', 'item_id', 'item_name', 'item_rank', 'item_score', 'iteration', 'critiqued_keyphrase', 'target_rank', 'num_existing_keyphrases', 'result', 'theta']

    df = pd.DataFrame(columns=columns)

    row = {}

    target_ranks = [1] # First get topk = 1, then extract other metrics
    # target_ranks = [20,50]

    # Randomly select test users
    # test_users = sample_users(num_users, num_users_sampled)
    # Test fixed users
    # test_users = [1]
    test_users = np.arange(50)
    
    # keyphrase_selection_method = "diff"
    max_wanted_keyphrase = 20
    # keyphrases_names = pd.read_csv('../data/yelp/KeyPhrases.csv')['Phrases'].tolist()
    
    print ("critiquing_model_name", critiquing_model_name)
    
    critiquing_model = critiquing_models[critiquing_model_name](keyphrase_freq=keyphrase_freq,
                                                                item_keyphrase_freq=item_keyphrase_freq,
                                                                row=row,
                                                                matrix_Train=matrix_Train,
                                                                matrix_Test=matrix_Test,
                                                                test_users=test_users,
                                                                target_ranks=target_ranks,
                                                                num_items_sampled=num_items_sampled,
                                                                num_keyphrases=num_keyphrases,
                                                                df=df,
                                                                max_iteration_threshold=max_iteration_threshold,
                                                                keyphrase_popularity=keyphrase_popularity,
                                                                dataset_name=dataset_name,
                                                                model=model,
                                                                parameters_row=parameters_row,
                                                                keyphrases_names = keyphrases_names,
                                                                keyphrase_selection_method = keyphrase_selection_method,
                                                                max_wanted_keyphrase = max_wanted_keyphrase,
                                                                lamb = lamb)

    df = critiquing_model.start_critiquing()

    return df

