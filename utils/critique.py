from gurobipy import *
from prediction.predictor import predict_scores
from utils.progress import inhour

import time
import numpy as np


def sample_users(num_users, num_users_sampled):
    return np.random.choice(num_users, num_users_sampled, replace=False)

def sample_items(candidate_items, num_items_sampled):
    return np.random.choice(candidate_items, size=num_items_sampled, replace=False)

def sample_keyphrase():
    # Critique the most popular keyphrases in the remaining
    critiqued_keyphrase = remaining_keyphrases[np.argmax(self.keyphrase_popularity[remaining_keyphrases])]

def get_max_length(df, num_keyphrases, max_iteration):
    df_s_f = df[(df['result'] == 'successful') | (df['result'] == 'fail')]
    df_s_f.loc[df_s_f['num_existing_keyphrases'] > max_iteration, 'num_existing_keyphrases'] = max_iteration
    return df_s_f['num_existing_keyphrases'].mean()

def get_average_length(df, n):
    df_s_f = df[(df['result'] == 'successful') | (df['result'] == 'fail')]
    iteration = df_s_f[df_s_f['target_rank']==n].groupby('user_id', as_index=False).agg({'iteration':'mean'})['iteration'].to_numpy()
    return (np.average(iteration), 1.96*np.std(iteration)/np.sqrt(len(iteration)))

def get_success_num(df, n):
    return len(df[(df['result'] == 'successful') & (df['target_rank'] == n)])

def get_fail_num(df, n):
    return len(df[(df['result'] == 'fail') & (df['target_rank'] == n)])

def get_success_rate(df, n):
    df_s_f = df[(df['result'] == 'successful') | (df['result'] == 'fail')]
    df_list_result = df_s_f[df_s_f['target_rank']==n].groupby('user_id', as_index=False)['result'].apply(list).reset_index(name='result')
    successful_rate = df_list_result['result'].apply(lambda r: r.count("successful")/len(r)).to_numpy()
    return (np.average(successful_rate), 1.96*np.std(successful_rate)/np.sqrt(len(successful_rate)))

def count_occurrence(x):
    return x.count("successful")

def add_pop(x, item_pop_index):
    return np.where(item_pop_index == x)[0][0]

def lpaverage(initial_prediction_u, keyphrase_freq, affected_items, unaffected_items, num_keyphrases, query, test_user, item_latent, reg, all_equal = True):
    critiqued_vector = np.zeros(keyphrase_freq.shape[1])
    
    for q in query:
        critiqued_vector[q] = max(keyphrase_freq[test_user][q],1)
#         critiqued_vector[q] = keyphrase_freq[test_user,q]
        
    num_critiques = len(query)
    
    # Get item latent for updating prediction
    W2 = reg.coef_
    W = item_latent.dot(W2)
    
    optimal_lambda = 1 # weight all critiquing equally
    lambdas = [optimal_lambda]*num_critiques
    
    # Record lambda values 
    for k in range(num_critiques):
        critiqued_vector[query[k]] *= optimal_lambda

    critique_score = predict_scores(matrix_U=reg.predict(critiqued_vector.reshape(1, -1)),
                                    matrix_V=item_latent)

    if all_equal:
        # weight initial and each critiquing equally 
        new_prediction = initial_prediction_u/(num_critiques) + critique_score.flatten()
    else:
        # weight intial and combined critiquing equally
        new_prediction = initial_prediction_u + critique_score.flatten() 
#     print (len(new_prediction))
    return new_prediction, lambdas   


def LP1SimplifiedOptimize(initial_prediction_u, keyphrase_freq, affected_items, unaffected_items, num_keyphrases, query, test_user, item_latent, reg):

    critiqued_vector = np.zeros(keyphrase_freq[0].shape)

    for q in query:
        # critiqued_vector[q] = -keyphrase_freq[test_user][q]
        critiqued_vector[q] = max(keyphrase_freq[test_user][q],1)

    num_critiques = len(query)

    W2 = reg.coef_
    W = item_latent.dot(W2)

    num_affected_items = len(affected_items)
    num_unaffected_items = len(unaffected_items)

    # start_time = time.time()

    # Model
    m = Model("LP1Simplified")
    m.setParam('OutputFlag', 0)
    # Assignment variables
    lambs = []

    for k in range(num_critiques):
        lambs.append(m.addVar(lb=-1,
                              ub=1,
                              vtype=GRB.CONTINUOUS,
                              name="lamb%d" % query[k]))

    m.setObjective(quicksum(initial_prediction_u[affected_item] * num_unaffected_items + quicksum(lambs[k] * critiqued_vector[query[k]] * W[affected_item][query[k]] * num_unaffected_items for k in range(num_critiques)) for affected_item in affected_items) - quicksum(initial_prediction_u[unaffected_item] * num_affected_items + quicksum(lambs[k] * critiqued_vector[query[k]] * W[unaffected_item][query[k]] * num_affected_items for k in range(num_critiques)) for unaffected_item in unaffected_items), GRB.MAXIMIZE)

    # Optimize
    m.optimize()

    # print("Elapsed: {}".format(inhour(time.time() - start_time)))

    lambdas = []
    for k in range(num_critiques):
        optimal_lambda = m.getVars()[k].X
        lambdas.append(optimal_lambda)
        critiqued_vector[query[k]] *= optimal_lambda

    critique_score = predict_scores(matrix_U=reg.predict(critiqued_vector.reshape(1, -1)),
                                    matrix_V=item_latent)

    new_prediction = initial_prediction_u + critique_score.flatten()

    return new_prediction, lambdas



def lpranksvm1(initial_prediction_u, keyphrase_freq, affected_items, unaffected_items, num_keyphrases, 
            query, test_user, item_latent, reg, user_latent_embedding, item_keyphrase_freq, Y):
    
    """
    See https://www.overleaf.com/read/wwftdhpcmxnx
    Section constraint generation technique
    """
    critiques = query # fix this variable name later
    num_critiques = len(critiques)

    W2 = reg.coef_
    W = item_latent.dot(W2)

    num_affected_items = len(affected_items)
    num_unaffected_items = len(unaffected_items)

    start_time = time.time()

    # Model
    m = Model("RankSVM1")
    m.setParam('OutputFlag', 0)
    
    # Assignment variables
    thetas = []
    us = []
    xis = []
    # weight thetas
    for k in range(num_critiques + 1):
        thetas.append(m.addVar(lb=-1,
                              ub=1,
                              vtype=GRB.CONTINUOUS,
                              name="theta%d" % k))
    thetas = np.array(thetas)
    # dummy variable u for absolute theta
    for k in range(num_critiques + 1):
        us.append(m.addVar(vtype=GRB.CONTINUOUS,
                          name="u%d" % k))
        
    # slack variables xi
    for i in range(num_affected_items):
        for j in range(num_unaffected_items):
            xis.append(m.addVar(lb = 0, 
                                vtype = GRB.CONTINUOUS,
                                name = "xi_%d_%d" % (i,j) ))
      
    ## constraints
    # constraints for dummy variable u's
    for k in range(num_critiques+1):
        m.addConstr(us[k] >= thetas[k])
        m.addConstr(us[k] >= -thetas[k])
        
    ## Pre-calculate critique embedding
    u_i = Y[test_user]
    phi_js = []
    phi_jprimes = []
    k_cis = []

    user_latent_embedding = np.array(user_latent_embedding)
#     print ('user latent embedding shape: ', user_latent_embedding.shape)
    # constraints for rankSVM 
    for j in range(num_affected_items):
        for j_ in range(num_unaffected_items):
            m.addConstr( thetas.dot(user_latent_embedding.dot(item_latent[affected_items[j]])) >= thetas.dot(user_latent_embedding.dot(item_latent[unaffected_items[j_]])) + 1 - xis[j*num_affected_items + j_], name = "constraints%d_%d" % (j,j_))

    lamb = 5 #regularization parameter (trading-off margin size against training error
    m.setObjective(quicksum(us) + lamb * quicksum(xis), GRB.MINIMIZE)
                
    # Optimize
    m.optimize()

#     print("Elapsed: {}".format(inhour(time.time() - start_time)))

    thetas = []
    for k in range(num_critiques+1):
        optimal_theta = m.getVarByName("theta%d" % k).X
        thetas.append(optimal_theta)
        
    critiqued_vector = np.zeros(keyphrase_freq.shape[1])
    
    # Combine weights to critiqued vector
    for c in critiques:
#         critiqued_vector[q] = 1 # set critiqued/boosted keyphrase to 1
        critiqued_vector[c] = max(keyphrase_freq[test_user , c],1)
    for k in range(num_critiques):
        critiqued_vector[critiques[k]] *= thetas[k+1]
    
    # Get rating score
    critique_score = predict_scores(matrix_U=reg.predict(critiqued_vector.reshape(1, -1)),
                                    matrix_V=item_latent)
    new_prediction = thetas[0]*initial_prediction_u + critique_score.flatten()
    
    return new_prediction, thetas




def lpranksvm2(initial_prediction_u, keyphrase_freq, affected_items, unaffected_items, num_keyphrases, 
            query, test_user, item_latent, reg, user_latent_embedding, item_keyphrase_freq, Y, lamb = [5,5]):
    """
    See https://www.overleaf.com/read/wwftdhpcmxnx
    """
    critiques = query # fix this variable name later

    # pre calculate some value
    num_critiques = len(critiques)

    num_affected_items = len(affected_items)
    num_unaffected_items = len(unaffected_items)

#     start_time = time.time()

    # Model
    m = Model("RankSVM2")
    m.setParam('OutputFlag', 0) # set to 1 for outputing details
    
    # Assignment variables
    thetas = []
    us = []
    xi_pos = []
    xi_neg = []
    # weight thetas
    for k in range(num_critiques + 1):
        thetas.append(m.addVar(lb=-1,
                              ub=1,
                              vtype=GRB.CONTINUOUS,
                              name="theta%d" % k))
    thetas = np.array(thetas)
    
    # dummy variable u for absolute theta
    for k in range(num_critiques + 1):
        us.append(m.addVar(vtype=GRB.CONTINUOUS,
                          name="u%d" % k))
        
    # slack variables xi
    for i in range(num_affected_items):
        xi_pos.append(m.addVar(lb = 0, 
                                vtype = GRB.CONTINUOUS,
                                name = "xi_pos%d" % i ))
    for i in range(num_unaffected_items):
        xi_neg.append(m.addVar(lb = 0, 
                                vtype = GRB.CONTINUOUS,
                                name = "xi_neg%d" % i ))
        
    ## constraints
    # constraints for dummy variable u's
    for k in range(num_critiques+1):
        m.addConstr(us[k] >= thetas[k])
        m.addConstr(us[k] >= -thetas[k])
 
    user_latent_embedding = np.array(user_latent_embedding)
    
    # Affected items rank higher
    for j in range(num_affected_items):
        m.addConstr( thetas.dot(user_latent_embedding.dot(item_latent[affected_items[j]])) >= initial_prediction_u[affected_items[j]] + 1 - xi_pos[j], name = "pos_constraint%d" % j )
    
    # Unaffected items rank lower
    for j in range(num_unaffected_items):
        m.addConstr( initial_prediction_u[unaffected_items[j]] - thetas.dot(user_latent_embedding.dot(item_latent[unaffected_items[j]])) >=  1 - xi_neg[j], name = "neg_constraint%d" % j )
            
    # objective
    lamb1 = lamb[0] #regularization for trading-off margin size against training error
    lamb2 = lamb[1] #regularization for trading-off deviation from Averaging 
    m.setObjective(quicksum(us) + lamb1 * (quicksum(xi_pos)+quicksum(xi_neg)) + lamb2 * quicksum( [( 1- theta) for theta in thetas]), GRB.MINIMIZE) 
                
    # Optimize
    m.optimize()

    # Save optimal thetas
    thetas = []
    for k in range(num_critiques+1):
        optimal_theta = m.getVarByName("theta%d" % k).X
        thetas.append(optimal_theta)
        
    critiqued_vector = np.zeros(keyphrase_freq.shape[1])
    
    # Combine weights to critiqued vector
    for c in critiques:
#         critiqued_vector[c] = 1 # set critiqued/boosted keyphrase to 1
        critiqued_vector[c] = max(keyphrase_freq[test_user , c],1)
    
    for k in range(num_critiques):
        critiqued_vector[critiques[k]] *= thetas[k+1]
    
    # Get rating score
    critique_score = predict_scores(matrix_U=reg.predict(critiqued_vector.reshape(1, -1)),
                                    matrix_V=item_latent)
    new_prediction = thetas[0]*initial_prediction_u/num_critiques + critique_score.flatten()
    
    return new_prediction, thetas


def lpranksvm3(initial_prediction_u, keyphrase_freq, affected_items, unaffected_items, num_keyphrases, 
            query, test_user, item_latent, reg, user_latent_embedding, item_keyphrase_freq, Y, lamb = [5,5]):
    critiques = query # fix this variable name later

    # pre calculate some value
    num_critiques = len(critiques)

    num_affected_items = len(affected_items)
    num_unaffected_items = len(unaffected_items)

#     start_time = time.time()

    # Model
    m = Model("LP2RankSVM3")
    m.setParam('OutputFlag', 0) # set to 1 for outputing details
    
    # Assignment variables
    thetas = []
    us = []
    xi_pos = []
    xi_neg = []
    # weight thetas
    for k in range(num_critiques + 1):
        thetas.append(m.addVar(lb=-2,
                              ub=2,
                              vtype=GRB.CONTINUOUS,
                              name="theta%d" % k))
    thetas = np.array(thetas)
    
    # dummy variable u for absolute theta
    for k in range(num_critiques + 1):
        us.append(m.addVar(vtype=GRB.CONTINUOUS,
                          name="u%d" % k))
        
    # slack variables xi
    for i in range(num_affected_items):
        xi_pos.append(m.addVar(lb = 0, 
                                vtype = GRB.CONTINUOUS,
                                name = "xi_pos%d" % i ))
    for i in range(num_unaffected_items):
        xi_neg.append(m.addVar(lb = 0, 
                                vtype = GRB.CONTINUOUS,
                                name = "xi_neg%d" % i ))
        
    ## constraints
    # constraints for dummy variable u's
    for k in range(num_critiques+1):
        m.addConstr(us[k] >= thetas[k] - 1)
        m.addConstr(us[k] >= 1 - thetas[k])
 
    user_latent_embedding = np.array(user_latent_embedding)
    
    # Affected items rank higher
    for j in range(num_affected_items):
        m.addConstr( thetas.dot(user_latent_embedding.dot(item_latent[affected_items[j]])) >= initial_prediction_u[affected_items[j]] + 1 - xi_pos[j], name = "pos_constraint%d" % j )
    
    # Unaffected items rank lower
    for j in range(num_unaffected_items):
        m.addConstr( initial_prediction_u[unaffected_items[j]] - thetas.dot(user_latent_embedding.dot(item_latent[unaffected_items[j]])) >=  1 - xi_neg[j], name = "neg_constraint%d" % j )
            
    # objective
    if type(lamb) != list:
        m.setObjective(quicksum(us) + lamb * (quicksum(xi_pos)+quicksum(xi_neg)), GRB.MINIMIZE)  # Single regularization
    else:
        lamb1 = lamb[0] #regularization for trading-off margin size against training error
        lamb2 = lamb[1] #regularization for trading-off deviation from Averaging 
        m.setObjective(lamb1* quicksum(us) + lamb2 * (quicksum(xi_pos)+quicksum(xi_neg)), GRB.MINIMIZE) # double regularization
    
                
    # Optimize
    m.optimize()

    # Save optimal thetas
    thetas = []
    for k in range(num_critiques+1):
        optimal_theta = m.getVarByName("theta%d" % k).X
        thetas.append(optimal_theta)
        
    critiqued_vector = np.zeros(keyphrase_freq.shape[1])
    
    # Combine weights to critiqued vector
    for c in critiques:
#         critiqued_vector[c] = 1 # set critiqued/boosted keyphrase to 1
        critiqued_vector[c] = max(keyphrase_freq[test_user , c],1)
    
    for k in range(num_critiques):
        critiqued_vector[critiques[k]] *= thetas[k+1]
    
    # Get rating score
    critique_score = predict_scores(matrix_U=reg.predict(critiqued_vector.reshape(1, -1)),
                                    matrix_V=item_latent)
    new_prediction = thetas[0]*initial_prediction_u/num_critiques + critique_score.flatten()
#     new_prediction = initial_prediction_u/num_critiques + critique_score.flatten()
#     new_prediction = critique_score.flatten()
    
    return new_prediction, thetas