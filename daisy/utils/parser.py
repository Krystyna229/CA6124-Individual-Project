import argparse
from json import dumps, loads, JSONDecodeError

import yaml

def parse_args():
    parser = argparse.ArgumentParser(description='arguments for daisy')
    # tuner settings
    parser.add_argument('--optimization_metric', 
                        type=str, 
                        help='the metric to be optimized for hyper-parameter tuning via HyperOpt')
    parser.add_argument('--optimization_k',
                        type=int,
                        help='optional cutoff used by the tuning objective')
    parser.add_argument('--hyperopt_trail', 
                        type=int, 
                        help='the number of trails of HyperOpt')
    parser.add_argument('--study_storage',
                        type=str,
                        help='optional Optuna storage URL for resumable tuning')
    parser.add_argument('--study_name',
                        type=str,
                        help='optional persistent Optuna study name')
    parser.add_argument('--tune_pack', 
                        type=str, 
                        help='record the searching space of hyper-parameters for HyperOpt')
    # common settings
    parser.add_argument('--algo_name', 
                        type=str, 
                        help='algorithm to choose')
    parser.add_argument('--dataset', 
                        type=str, 
                        help='select dataset')
    parser.add_argument('--prepro', 
                        type=str, 
                        help='dataset preprocess op.: origin/Nfilter/Ncore')
    parser.add_argument('--topk', 
                        type=int, 
                        help='top number of recommend list')
    parser.add_argument('--test_method', 
                        type=str, 
                        help='method for split test,options: tsbr/rsbr/tloo/rloo')
    parser.add_argument('--val_method', 
                        type=str, 
                        help='validation method, options: tsbr/rsbr/tloo/rloo')
    parser.add_argument('--test_size', 
                        type=float, 
                        help='split ratio for test set')
    parser.add_argument('--val_size', 
                        type=float, 
                        help='split ratio for validation set')
    parser.add_argument('--fold_num', 
                        type=int, 
                        help='No. of folds for cross-validation')
    parser.add_argument('--cand_num', 
                        type=int, 
                        help='the number of candidate items used for ranking')
    parser.add_argument('--ranking_mode',
                        type=str,
                        help='ranking protocol: sampled (original) or full')
    parser.add_argument('--warm_start',
                        action='store_true',
                        default=None,
                        help='evaluate only users and items observed in training')
    parser.add_argument('--split_boundary',
                        type=str,
                        help='timestamp boundary: interaction (original) or day')
    parser.add_argument('--sample_method', 
                        type=str, 
                        help='negative sampling method mixed with uniform, options: high-pop, low-pop')
    parser.add_argument('--sample_ratio', 
                        type=float, 
                        help='control the ratio of popularity sampling for the hybrid sampling strategy in the range of (0,1)')
    parser.add_argument('--init_method', 
                        type=str, 
                        default='default', 
                        help='weight initialization method: normal, uniform, xavier_normal, xavier_uniform')
    parser.add_argument('--gpu', 
                        type=str, 
                        help='gpu card ID')
    parser.add_argument('--num_ng', 
                        type=int, 
                        help='negative sampling number')
    parser.add_argument('--loss_type', 
                        type=str, 
                        help='loss function type')
    parser.add_argument('--optimizer', 
                        type=str, 
                        help='optimize method')
    # algo settings
    parser.add_argument('--factors', 
                        type=int, 
                        help='latent factors numbers in the model')
    parser.add_argument('--latent_dim', 
                        type=int, 
                        help='bottleneck layer size for autoencoder')
    parser.add_argument('--context_window', 
                        type=int, 
                        help='the one-side window size of skip-gram')
    parser.add_argument('--rho', 
                        type=float, 
                        help='discard threshold for sequence in item2vec')
    parser.add_argument('--reg', 
                        type=float, 
                        help='EASE regularization term')        
    parser.add_argument('--reg_1', 
                        type=float, 
                        help='L1 regularization')
    parser.add_argument('--anneal_cap', 
                        type=float, 
                        help='Anneal penalty for VAE KL loss')
    parser.add_argument('--reg_2', 
                        type=float, 
                        help='L2 regularization')
    parser.add_argument('--alpha', 
                        type=float, 
                        help='constant to multiply the penalty terms for SLIM')
    parser.add_argument('--elastic', 
                        type=float, 
                        help='the ElasticNet mixing parameter for SLIM in the range of (0,1)')
    parser.add_argument('--maxk', 
                        type=int, 
                        help='The (max) number of neighbors to take into account')
    parser.add_argument('--node_dropout',
                        type=float,  
                        help='NGCF: Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--mess_dropout', 
                        type=float, 
                        help='NGCF: Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--dropout', 
                        type=float, 
                        help='dropout rate')
    parser.add_argument('--lr', 
                        type=float, 
                        help='learning rate')
    parser.add_argument('--epochs', 
                        type=int, 
                        help='training epochs')
    parser.add_argument('--batch_size', 
                        type=int, 
                        help='batch size for training')
    parser.add_argument('--loader_workers',
                        type=int,
                        help='number of DataLoader worker processes')
    parser.add_argument('--num_layers', 
                        type=int, 
                        help='number of layers in MLP model')
    parser.add_argument('--act_function', 
                        type=str, 
                        default='relu', 
                        help='activation method in interio layers')
    parser.add_argument('--batch_norm', 
                        action='store_true', 
                        help='whether do batch normalization in interior layers')
    parser.add_argument("--early_stop",
                        action="store_true",
                        help="whether to activate early stop mechanism")
    args = parser.parse_args()

    return args



def json_format_corrector(json_string: str) -> str:
    '''
    Argparse may not parse command line arguments with double quotes
    Takes the string produced from the command line parser, and corrects the string format to fit json.loads()
    Adds a double quote to all keys and non-numeric values (except null)
    '''
    
    # Check if the string is empty (or is None)
    if not json_string or json_string == '{}': return json_string

    # Keep valid JSON unchanged.
    try:
        loads(json_string)
        return json_string
    except JSONDecodeError:
        pass

    # DaisyRec's command-line examples use YAML-style mappings with unquoted
    # keys. Safe YAML parsing handles those mappings without corrupting keys
    # such as ``reg_1`` (which the original character-level converter did).
    parsed = yaml.safe_load(json_string)
    if not isinstance(parsed, dict):
        raise ValueError('tune_pack must be a JSON/YAML mapping')
    return dumps(parsed)
