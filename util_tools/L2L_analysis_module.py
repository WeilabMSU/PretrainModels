'''
Analytic Hierarchy Process, AHP.
Base on Wasserstein distance
'''


from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
import scipy
import numpy as np
import pandas as pd
import sys
import argparse
import os
import glob
import datasets_analysis_module as dam


class idx_analysis(object):
    def __init__(self):
        self.all_distribution_idx = {
            'c': 0, 'C': 1, '(': 2, ')': 3, '1': 4, 'O': 5, '=': 6, '2': 7, 'N': 8, 'n': 9,
            '3': 10, '[': 11, ']': 12, '@': 13, 'H': 14, 'F': 15, '-': 16, '4': 17, 'S': 18, 'Cl': 19,
            '/': 20, 's': 21, 'o': 22, '.': 23, 'Br': 24, '5': 25, '+': 26, '#': 27, '\\': 28, '6': 29,
            'I': 30, 'P': 31, 'Si': 32, '7': 33, '8': 34, 'B': 35, '%': 36, 'Na': 37, '9': 38, '0': 39,
            'K': 40, 'Sn': 41, 'Se': 42, 'Li': 43, 'Zn': 44, 'Al': 45, 'b': 46, 'As': 47, 'Mg': 48, 'p': 49,
            'Ca': 50, 'se': 51, 'Ag': 52, 'Te': 53, 'Ba': 54, 'Bi': 55, 'Rb': 56, 'Cs': 57, 'Sr': 58, 'te': 59,
            'Be': 60, 'length': 61, 'symbol_type': 62
        }
        self.all_distribution_idx_reversed = {v: k for k, v in self.all_distribution_idx.items()}


def wasserstein_dis(distr_dict_0, distr_dict_1, dis_type='wasserstein'):
    minus = 1e-15
    sorted_keys_0 = np.sort(list(distr_dict_0.keys()))
    max_value_0 = max(distr_dict_0.values())
    values_0 = minus + np.array([distr_dict_0[k] for k in sorted_keys_0])/max_value_0

    sorted_keys_1 = np.sort(list(distr_dict_1.keys()))
    max_value_1 = max(distr_dict_1.values())
    values_1 = minus + np.array([distr_dict_1[k] for k in sorted_keys_1])/max_value_1

    if dis_type == 'wasserstein':
        w_dis = wasserstein_distance(values_0, values_1)
    elif dis_type == 'KL':
        w_dis = np.mean(scipy.special.kl_div(values_0, values_1))
    else:
        w_dis = np.linalg.norm(np.array(values_0) - np.array(values_1))

    return np.round(w_dis, 4)


def datasets_pair_analysis(
    target_set_distribution,
    pretrain_sets_distribution_path='PretrainedSetsDistribution.npy'
):
    if not os.path.exists(pretrain_sets_distribution_path):
        print(pretrain_sets_distribution_path, 'not the right file.')
        print('PretrainedSetsDistribution.npy can not be found')
    pretrained_sets_distribution = np.load(pretrain_sets_distribution_path, allow_pickle=True).item()
    three_sets_prefix = ['c', 'cp', 'cpz']

    all_wd_values = {k: {} for k in three_sets_prefix}
    for i, prefix in enumerate(three_sets_prefix):
        for j in range(63):
            prefix_name = f"{prefix}-{j}"
            all_wd_values[prefix][j] = wasserstein_dis(
                target_set_distribution[str(j)],
                pretrained_sets_distribution[prefix_name]
            )
    return all_wd_values


def rerange_distribution(target, combined_result):
    distribute_dict = {}
    if target == 'length':
        min_len, max_len = 1, 256
        distribute_dict = {k: 0 for k in range(min_len, max_len+1)}
        for k, v in combined_result.items():
            if k <= min_len:
                distribute_dict[min_len] += v
            elif k > min_len and k < max_len:
                distribute_dict[k] = v
            elif k >= max_len:
                distribute_dict[max_len] += v
            else:
                print('Unexpected key from combined_result.(target: length)')
    elif target == 'symbol_type':
        min_len, max_len = 1, 61
        distribute_dict = {k: 0 for k in range(min_len, max_len+1)}
        for k, v in combined_result.items():
            if k <= min_len:
                distribute_dict[min_len] += v
            elif k > min_len and k < max_len:
                distribute_dict[k] = v
            elif k >= max_len:
                distribute_dict[max_len] += v
            else:
                print('Unexpected key from combined_result.(target: symbol_type)')
    else:
        distribute_dict = {k: 0 for k in [np.round(w, 2) for w in np.arange(0.0, 1.001, 0.01)]}
        for k, v in combined_result.items():
            if k in distribute_dict:
                distribute_dict[k] += v
            else:
                print('Unexpected key {:s} from combined_result.(consider_symbol {:s})'.format(str(k), target))
    return distribute_dict


def linear_ridgeclassifier(x, y):
    from sklearn import linear_model
    cla = linear_model.RidgeClassifier()
    cla.fit(x, y)
    return cla.score(x, y), cla.intercept_, cla


def data_norm(*args):
    assert len(args) > 0, "Datasets' length needs > 0"
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(np.vstack(args))
    norm_args = [scaler.transform(args[i]) for i in range(len(args))]
    norm_args = norm_args if len(args) > 1 else norm_args[0]
    return norm_args


def main_get_dis_customized_dataset(file='./temp_data/bbbp.smi', num_workers=1):
    # savename = 'wasserstein_temp.csv'
    dataname = os.path.split(file)[-1].split('.')[0]

    ahp = idx_analysis()
    all_features = []

    target_set_distribution = {}
    for k, v in ahp.all_distribution_idx.items():
        ta = dam.target_analysis(k)
        if k == 'length':
            specific_func = ta.length_analysis
        elif k == 'symbol_type':
            specific_func = ta.symbol_type_analysis
        else:
            specific_func = ta.symbol_analysis
        combined_result = dam.parallel_operation(file, num_workers, specific_func)
        distribute_dict = rerange_distribution(k, combined_result)
        target_set_distribution[str(v)] = distribute_dict
    all_wd_values = datasets_pair_analysis(
        target_set_distribution,
        pretrain_sets_distribution_path='PretrainedSetsDistribution.npy',
    )

    # 3 to 1
    for nd, (k, wd_dict) in enumerate(all_wd_values.items()):
        all_features.append(list(wd_dict.values()))

    final_features = pd.DataFrame(
        np.reshape(all_features, [1, 63*3]),  # (all_features),
        index=[dataname],
        columns=list(range(63*3)),
    )
    # final_features.to_csv(savename)
    return final_features


def main_L2L(args):
    filename = './wasserstein.csv'  # This file contains the features used to train the decision model.
    if not os.path.exists(filename):
        print('No wasserstein.csv exists')

    data_df = pd.read_csv(filename, header=0, index_col=0)
    label = data_df['label'].values
    features = data_df[[str(i) for i in range(np.shape(data_df.values)[-1]-1)]].values
    # print(features.shape)

    customized_dataset_feature = main_get_dis_customized_dataset(
        file=args.input_dataset, num_workers=args.num_workers).values
    all_features = np.vstack([features, customized_dataset_feature])
    norm_all_features = data_norm(all_features)
    features = norm_all_features[0: -1, :]
    customized_dataset_feature = norm_all_features[-1, :]

    all_score = []
    all_inter = []
    flag = 1
    for redu_i in range(1, np.shape(features)[0]+1):
        reducer = PCA(n_components=redu_i)
        features_ = reducer.fit_transform(features)
        score, inter_, model = linear_ridgeclassifier(features_, label)
        all_score.append(score)
        all_inter.append(inter_[0])
        # print(redu_i, score)

        if score - 1 == 0 and flag == 1:
            customized_dataset_feature_ = reducer.transform(customized_dataset_feature[None, :])
            get_scores = model.decision_function(customized_dataset_feature_)
            # print(model.decision_function(features_))
            flag = 0

    # print(all_score)
    # print(all_inter)
    select_models = {0: 'model_chembl27', 1: 'model_chembl27_pubchem', 2: 'model_chembl27_pubchem_zinc'}
    print(f'Select the pretrained {select_models[np.argmax(get_scores)]}, and the score is {np.max(get_scores)}')


def main(args):
    main_L2L(args)


def parse_args(args):
    parser = argparse.ArgumentParser(description='Datasets analysis')
    parser.add_argument('--input_dataset', default='test.smi', type=str)
    parser.add_argument('--num_workers', default=1, type=int)

    args = parser.parse_args()
    return args


def cli_main():
    args = parse_args(sys.argv[1:])
    # print(args)
    main(args)


if __name__ == "__main__":
    cli_main()
    print('End!')
