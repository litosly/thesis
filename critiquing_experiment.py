from experiment.critiquing import critiquing
from utils.argcheck import check_int_positive
from utils.io import load_numpy, load_yaml, load_dataframe_csv, save_dataframe_csv, find_best_hyperparameters
from utils.modelname import models
from utils.progress import WorkSplitter

import argparse


def main(args):
    # Progress bar
    progress = WorkSplitter()

    # Show hyperparameter settings
    progress.section("Parameter Setting")
    print("Data Directory: {}".format(args.data_dir))
    print("Number of Users Sampled: {}".format(args.num_users_sampled))
    print("Number of Items Sampled: {}".format(args.num_items_sampled))
    print("Number of Max Allowed Iterations: {}".format(args.max_iteration_threshold))
    print("Critiquing Model: {}".format(args.critiquing_model_name))

    R_train = load_numpy(path=args.data_dir, name=args.train_set)
    print("Train U-I Dimensions: {}".format(R_train.shape))

    R_test = load_numpy(path=args.data_dir, name=args.test_set)
    print("Test U-I Dimensions: {}".format(R_test.shape))

    R_train_keyphrase = load_numpy(path=args.data_dir, name=args.train_keyphrase_set).toarray()
    print("Train Item Keyphrase U-I Dimensions: {}".format(R_train_keyphrase.shape))

    R_train_item_keyphrase = load_numpy(path=args.data_dir, name=args.train_item_keyphrase_set).toarray()

    table_path = load_yaml('config/global.yml', key='path')['tables']
    # parameters = find_best_hyperparameters(table_path+args.dataset_name, 'NDCG')
    # parameters_row = parameters.loc[parameters['model'] == args.model]

    parameters_row = {
        'iter' : 10,
        'lambda' : 200,
        'rank' : 200 
    }
    
    keyphrases_names = load_dataframe_csv(path = args.data_dir, name = "Keyphrases.csv")['Phrases'].tolist()
    
    results = critiquing(matrix_Train=R_train,
                         matrix_Test=R_test,
                         keyphrase_freq=R_train_keyphrase,
                         item_keyphrase_freq=R_train_item_keyphrase.T,
                         num_users_sampled=args.num_users_sampled,
                         num_items_sampled=args.num_items_sampled,
                         max_iteration_threshold=args.max_iteration_threshold,
                         dataset_name=args.dataset_name,
                         model=models[args.model],
                         parameters_row=parameters_row,
                         critiquing_model_name=args.critiquing_model_name,
                         lamb = args.lambdas,
                         keyphrases_names = keyphrases_names,
                         keyphrase_selection_method = args.keyphrase_selection_method)

    table_path = load_yaml('config/global.yml', key='path')['tables']
    save_dataframe_csv(results, table_path, args.save_path)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Latent Linear Critiquing")

    parser.add_argument('--critiquing_model_name', dest='critiquing_model_name', default="ranksvm3",
                        help='Critiquing model. (default: %(default)s)')

    parser.add_argument('--data_dir', dest='data_dir', default="../data/yelp/",
                        help='Directory path to the dataset. (default: %(default)s)')

    parser.add_argument('--dataset_name', dest='dataset_name', default="yelp",
                        help='Dataset name. (default: %(default)s)')

    parser.add_argument('--model', dest='model', default="PLRec",
                        help='Model currently using. (default: %(default)s)')

    parser.add_argument('--save_path', dest='save_path', default="critiquing/tuning_using_py/test.csv",
                        help='Results saved path. (default: %(default)s)')

    parser.add_argument('--max_iteration_threshold', dest='max_iteration_threshold', default=20,
                        type=check_int_positive,
                        help='Maximum iterations allowed for each critiquing session. (default: %(default)s)')

    parser.add_argument('--num_items_sampled', dest='num_items_sampled', default=5,
                        type=check_int_positive,
                        help='Number of items sampled for each user in critiquing. (default: %(default)s)')

    parser.add_argument('--num_users_sampled', dest='num_users_sampled', default=25,
                        type=check_int_positive,
                        help='Number of users sampled in critiquing. (default: %(default)s)')

    parser.add_argument('--test', dest='test_set', default="Rtest.npz",
                        help='Test set sparse matrix. (default: %(default)s)')

    parser.add_argument('--train', dest='train_set', default="Rtrain.npz",
                        help='Train set sparse matrix. (default: %(default)s)')

    parser.add_argument('--train_keyphrase', dest='train_keyphrase_set', default="U_K.npz",
                        help='Train keyphrase sparse matrix. (default: %(default)s)')

    parser.add_argument('--train_item_keyphrase', dest='train_item_keyphrase_set', default="I_K.npz",
                        help='Train item keyphrase sparse matrix. (default: %(default)s)')

    parser.add_argument('--lambdas', dest='lambdas', default= [1, 1],
                        help='Lambdas for RankSVM. (default: %(default)s)')
    
    parser.add_argument('--keyphrase_selection_method', dest='keyphrase_selection_method', default="random",
                        help='keyphrase_selection_method. (default: %(default)s)')

    args = parser.parse_args()

    main(args)
