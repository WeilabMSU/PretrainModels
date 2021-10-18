"""
Tools kit for downstream jobs
"""


import torch
import numpy as np
import argparse
import os
import sys
from fairseq.models.roberta import RobertaModel


def prepare_input_data(pretrain_model, target_file):

    sample_num = 0
    for i, line in enumerate(open(target_file)):
        if len(line.strip()) == 0:
            continue
        sample_num += 1
    input_seq = np.ones([sample_num, pretrain_model.args.max_positions])

    for i, line in enumerate(open(target_file)):
        if len(line.strip()) == 0:
            continue
        tokens = pretrain_model.encode(line.strip())
        if len(tokens) > pretrain_model.args.max_positions:
            tokens = torch.cat(
                (tokens[:pretrain_model.args.max_positions - 1], tokens[-1].unsqueeze(0)))

        input_seq[i, 0: len(tokens)] = tokens

    return input_seq


def arange_hidden_info(pretrain_model, target_file, hidden_info):
    ''' arange_hidden_info for symbols specific features'''

    sample_num = 0
    for i, line in enumerate(open(target_file)):
        if len(line.strip()) == 0:
            continue
        sample_num += 1
    dict = pretrain_model.task.source_dictionary.symbols
    print(f'Dict: {dict}')
    dict_size = 54  # len(dict)
    arange_features = np.zeros(
        [sample_num, dict_size, pretrain_model.model.args.encoder_embed_dim])

    for i, line in enumerate(open(target_file)):
        if len(line.strip()) == 0:
            continue
        tokens = pretrain_model.encode(line.strip())
        if len(tokens) > pretrain_model.args.max_positions:
            tokens = torch.cat(
                (tokens[:pretrain_model.args.max_positions - 1], tokens[-1].unsqueeze(0)))
        # tokens shape [1, tokens_len]

        for dict_n in range(dict_size):
            location = torch.where(tokens == dict_n)[-1]
            if len(location) == 0:
                continue
            hidden = hidden_info[i]
            # hidden shape [tokens, embed_dim]

            arange_features[i, dict_n, :] = np.mean(hidden[location], axis=0)

    arange_features = np.reshape(arange_features, (sample_num, -1))

    return arange_features


def get_reconstracted_rate(pretrain_model, target_file):

    sample_num = 0
    for i, line in enumerate(open(target_file)):
        if len(line.strip()) == 0:
            continue
        sample_num += 1

    ncorrect = 0
    for i, line in enumerate(open(target_file)):
        if len(line.strip()) == 0:
            continue
        tokens = pretrain_model.encode(line.strip())
        if len(tokens) > pretrain_model.args.max_positions:
            tokens = torch.cat(
                (tokens[:pretrain_model.args.max_positions - 1], tokens[-1].unsqueeze(0)))

        out_layer, _ = pretrain_model.model(tokens.unsqueeze(0), features_only=False)
        pred = out_layer.argmax(-1)

        if (pred == tokens).all():
            ncorrect += 1
    print('Reconstructed rate: ' + str(ncorrect / float(sample_num)))

    return None


def extract_hidden(pretrain_model, target_file, args):

    sample_num = 0
    for i, line in enumerate(open(target_file)):
        if len(line.strip()) == 0:
            continue
        sample_num += 1
    hidden_features = {i: None for i in range(sample_num)}

    for i, line in enumerate(open(target_file)):
        if len(line.strip()) == 0:
            continue
        tokens = pretrain_model.encode(line.strip())
        if len(tokens) > pretrain_model.args.max_positions:
            tokens = torch.cat(
                (tokens[:pretrain_model.args.max_positions - 1], tokens[-1].unsqueeze(0)))

        _, all_layer_hiddens = pretrain_model.model(
            tokens.unsqueeze(0), features_only=True, return_all_hiddens=True)

        hidden_info = all_layer_hiddens['inner_states'][args.target_hidden]
        # last_hidden shape [tokens_num, sample_num(default=1), hidden_dim]

        # hidden_features.append(hidden_info.squeeze(1).cpu().detach().numpy())
        hidden_features[i] = hidden_info.squeeze(1).cpu().detach().numpy()

    # hidden_features type: dict, length: samples_num
    return hidden_features


def extract_features_from_hidden(args, hidden_info):

    extract_method = args.extract_features_method

    samples_num = len(hidden_info)
    hidden_dim = np.shape(hidden_info[0])[-1]
    if extract_method == 'average_all':
        samples_features = np.zeros([samples_num, hidden_dim])
        for n_sample, hidden in hidden_info.items():
            # hidden shape [tokens, embed_dim]
            samples_features[n_sample, :] = np.mean(hidden, axis=0)
    elif extract_method == 'bos':
        samples_features = np.zeros([samples_num, hidden_dim])
        for n_sample, hidden in hidden_info.items():
            # hidden shape [tokens, embed_dim]
            samples_features[n_sample, :] = hidden[0, :]
    elif extract_method == 'umap':
        import umap
        reducer = umap.UMAP(n_components=1)
        samples_features = np.zeros([samples_num, hidden_dim])
        for n_sample, hidden in hidden_info.items():
            scaled_penguin_data = np.array(hidden).T  # shape [embed_dim, tokens]
            embedding = reducer.fit_transform(scaled_penguin_data)
            print(embedding.shape)
            samples_features[n_sample, :] = embedding.flatten()

    return samples_features


def load_pretrain_model(model_name_or_path, checkpoint_file, data_name_or_path, bpe='smi'):
    '''Currently only load to cpu()'''

    # load model
    pretrain_model = RobertaModel.from_pretrained(
        model_name_or_path,
        checkpoint_file,
        data_name_or_path,  # dict_dir,
        bpe='smi',
    )
    pretrain_model.eval()
    return pretrain_model


def parse_args(args):
    parser = argparse.ArgumentParser(description="Tools kit for downstream jobs")

    parser.add_argument('--load_pretrain', default=False, action='store_true')
    parser.add_argument('--model_name_or_path', default=None, type=str,
                        help='Pretrained model folder')
    parser.add_argument('--checkpoint_file', default='checkpoint_best.pt', type=str,
                        help='Pretrained model name')
    parser.add_argument('--data_name_or_path', default=None, type=str,
                        help="Pre-training dataset folder")
    parser.add_argument('--bpe', default='smi', type=str)
    parser.add_argument('--target_file', default=None, type=str,
                        help="Target file for feature extraction, default format is .smi")
    parser.add_argument('--get_hidden_info', default=False, action='store_true')
    parser.add_argument('--get_hidden_info_from_model', default=False, action='store_true')
    parser.add_argument('--get_hidden_info_from_file', default=False, action='store_true')
    parser.add_argument('--get_reconstracted_rate', default=False, action='store_true')
    parser.add_argument('--save_hidden_info_path', default='hidden_info.npy', type=str)
    parser.add_argument('--hidden_info_path', default='hidden_info.npy', type=str)
    parser.add_argument('--target_hidden', default=-1, type=int,
                        help='Target hidden layer to extract features.')
    parser.add_argument('--extract_features_method', default='average_all', type=str,
                        help='select from [average_all, bos]')
    parser.add_argument('--extract_features_from_hidden_info', default=False, action='store_true')
    parser.add_argument('--save_feature_path', default='extract_f1.npy', type=str,
                        help="Saving feature filename(path)")
    parser.add_argument('--arange_hidden_info_features', default=False, action='store_true')
    parser.add_argument('--save_arange_features_path', default='arange_features.npy', type=str,
                        help='Arange hidden info, for symbol specific features')
    args = parser.parse_args()
    return args


def main(args):
    if args.load_pretrain:
        pretrain_model = load_pretrain_model(
            args.model_name_or_path, args.checkpoint_file, args.data_name_or_path, args.bpe)

    if args.get_hidden_info:
        if args.get_hidden_info_from_model:
            hidden_info = extract_hidden(pretrain_model, args.target_file, args)
            np.save(args.save_hidden_info_path, hidden_info)

        if args.get_hidden_info_from_file:
            assert os.path.exists(args.hidden_info_path), "Hidden info not exists"
            print(f'Hidden info loaded from {args.hidden_info_path}')
            hidden_info = np.load(args.hidden_info_path, allow_pickle=True).item()

    if args.extract_features_from_hidden_info:
        print('Generate features from hidden information')
        samples_features = extract_features_from_hidden(args, hidden_info)
        print(f'Features shape: {np.shape(samples_features)}')
        np.save(args.save_feature_path, samples_features)

    if args.arange_hidden_info_features:
        print('Arange hidden info for symbols specific features')
        assert args.load_pretrain, "Must load pretrain model"
        assert os.path.exists(args.target_file), "Target file not exists"
        arange_features = arange_hidden_info(
            pretrain_model, args.target_file, hidden_info)
        np.save(args.save_arange_features_path, arange_features)

    if args.get_reconstracted_rate:
        get_reconstracted_rate(pretrain_model, args.target_file)


def cli_main():
    args = parse_args(sys.argv[1:])
    print(args)
    main(args)


if __name__ == '__main__':
    cli_main()
    print('End!')

    # example code
    # python "/gpfs/wscgpfs02/chendo11/workspace/matai/fairseq_pro/utils_main.py" --load_pretrain --model_name_or_path /gpfs/wscgpfs01/chendo11/test/bos_finetune_from_chembl26all/finetune_info_from_bos_finetune/bos_finetune_train_4_data_seperate_5000_updates/bos_finetune_IGC50/ --checkpoint_file checkpoint_best.pt --data_name_or_path /gpfs/wscgpfs02/chendo11/workspace/matai/Dataset/finetune_data/IGC50/ --bpe smi --target_file "/gpfs/wscgpfs02/chendo11/workspace/matai/Dataset/finetune_data/IGC50/IGC50_test_canonical.smi" --save_regression_predict_path /gpfs/wscgpfs01/chendo11/test/bos_finetune_from_chembl26all/finetuned_direct_result/IGC50_predict_bos_finetune.npy --regression_label --regression_label_path "/gpfs/wscgpfs02/chendo11/workspace/matai/Dataset/finetune_data/IGC50/IGC50_test_y.npy"

    # get reconstract rate
    # python "/mnt/ufs18/home-180/chendo11/work_projects/BERT_comapre/fairseq_pro/utils_main.py" --load_pretrain --model_name_or_path "/mnt/ufs18/home-180/chendo11/work_projects/BERT_comapre/Pretrained_models/chembl27_512/" --checkpoint_file checkpoint_best.pt --data_name_or_path /mnt/ufs18/home-180/chendo11/work_projects/BERT_comapre/Pretrained_models/chembl27_512/ --bpe smi --target_file "/mnt/ufs18/home-180/chendo11/work_projects/BERT_comapre/Result_bert_c/Classification_finetune/HIV/seed_1818_split/train.smi" --get_reconstracted_rate
    
