'''
    Datasets analysis module
    - Input "smiles.smi" datasets.
    - Output distributions.
'''


import numpy as np
import time
from multiprocessing import Pool
import os
import sys
import argparse
import re


SPACE_NORMALIZER = re.compile(r"\s+")
SMI_SYMBOLS = r"Li|Be|Na|Mg|Al|Si|Cl|Ca|Zn|As|Se|se|Br|Rb|Sr|Ag|Sn|Te|te|Cs|Ba|Bi|[\d]|" + r"[HBCNOFPSKIbcnops#%\)\(\+\-\\\/\.=@\[\]]"


class target_analysis(object):
    def __init__(self, target='length'):
        self.target = target

    def length_analysis(self, smiles):
        return len(re.findall(SMI_SYMBOLS, smiles))

    def symbol_type_analysis(self, smiles):
        return len(list(set(re.findall(SMI_SYMBOLS, smiles))))

    def symbol_analysis(self, smiles):
        smi_list = re.findall(SMI_SYMBOLS, smiles)
        return np.round(smi_list.count(self.target)/len(smi_list), 2)


# split find chunk size
def find_offsets(filename, num_chunks):
    '''
        filename: input smiles file
        num_chunks: number of workers
    '''

    with open(filename, 'r', encoding='utf-8') as f:
        size = os.fstat(f.fileno()).st_size
        chunk_size = size // num_chunks
        offsets = [0 for _ in range(num_chunks + 1)]
        for i in range(1, num_chunks):
            f.seek(chunk_size * i)
            safe_readline(f)
            offsets[i] = f.tell()
        return offsets


def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins


def specific_operator(input_file, specific_func, offset=0, end=-1):

    result_dict = {}
    with open(input_file, 'r', encoding='utf-8') as f:
        f.seek(offset)
        line = safe_readline(f)
        while line:
            if end > 0 and f.tell() > end:
                break
            try:
                conted_result = specific_func(line.strip())
                if conted_result in result_dict:
                    result_dict[conted_result] += 1
                else:
                    result_dict[conted_result] = 1
            except:
                print(line)
            line = f.readline()
    return result_dict


def parallel_operation(input_file, num_workers, specific_func):

    combined_result = {}

    def count_result(worker_result):
        for k, v in worker_result.items():
            if k in combined_result:
                combined_result[k] += v
            else:
                combined_result[k] = v

    offsets = find_offsets(input_file, num_workers)
    pool = None
    if num_workers > 1:
        pool = Pool(processes=num_workers - 1)
        for worker_id in range(1, num_workers):
            pool.apply_async(
                specific_operator,
                (input_file, specific_func, offsets[worker_id], offsets[worker_id + 1]),
                callback=count_result,
            )
        pool.close()

    count_result(specific_operator(input_file, specific_func, offset=0, end=offsets[1]))

    if num_workers > 1:
        pool.join()

    # print(f'| {len(list(combined_result.keys()))} keys |')
    return combined_result


def statistic_one_operation(input_file, offset=0, end=-1):

    statistic_dict = {}
    with open(input_file, 'r', encoding='utf-8') as f:
        f.seek(offset)
        line = safe_readline(f)
        while line:
            if end > 0 and f.tell() > end:
                break
            try:
                conted_result = re.findall(SMI_SYMBOLS, line.strip())
                for sym in conted_result:
                    if sym in statistic_dict:
                        statistic_dict[sym] += 1
                    else:
                        statistic_dict[sym] = 1
            except:
                print(line)
            line = f.readline()
    return statistic_dict


def statistic_operation(input_file, num_workers):

    combined_result = {}

    def count_result(worker_result):
        for k, v in worker_result.items():
            if k in combined_result:
                combined_result[k] += v
            else:
                combined_result[k] = v

    offsets = find_offsets(input_file, num_workers)
    pool = None
    if num_workers > 1:
        pool = Pool(processes=num_workers - 1)
        for worker_id in range(1, num_workers):
            pool.apply_async(
                statistic_one_operation,
                (input_file, offsets[worker_id], offsets[worker_id+1]),
                callback=count_result,
            )
        pool.close()
    count_result(
        statistic_one_operation(input_file, offset=0, end=offsets[1]))
    if num_workers > 1:
        pool.join()

    return combined_result


def rerange_result_dict(combined_result, consider_symbols, args):
    distribute_dict = {}
    if args.target == 'length':
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
    elif args.target == 'symbol_type':
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
    elif args.target in consider_symbols:
        distribute_dict = {k: 0 for k in [np.round(w, 2) for w in np.arange(0.0, 1.001, 0.01)]}
        for k, v in combined_result.items():
            if k in distribute_dict:
                distribute_dict[k] += v
            else:
                print('Unexpected key {:s} from combined_result.(consider_symbol {:s})'.format(str(k), args.target))
    else:
        print(f'Wrong target {args.target}')

    all_targets = {
        'c': 0, 'C': 1, '(': 2, ')': 3, '1': 4, 'O': 5, '=': 6, '2': 7, 'N': 8, 'n': 9,
        '3': 10, '[': 11, ']': 12, '@': 13, 'H': 14, 'F': 15, '-': 16, '4': 17, 'S': 18, 'Cl': 19,
        '/': 20, 's': 21, 'o': 22, '.': 23, 'Br': 24, '5': 25, '+': 26, '#': 27, '\\': 28, '6': 29,
        'I': 30, 'P': 31, 'Si': 32, '7': 33, '8': 34, 'B': 35, '%': 36, 'Na': 37, '9': 38, '0': 39,
        'K': 40, 'Sn': 41, 'Se': 42, 'Li': 43, 'Zn': 44, 'Al': 45, 'b': 46, 'As': 47, 'Mg': 48, 'p': 49,
        'Ca': 50, 'se': 51, 'Ag': 52, 'Te': 53, 'Ba': 54, 'Bi': 55, 'Rb': 56, 'Cs': 57, 'Sr': 58, 'te': 59,
        'Be': 60, 'length': 61, 'symbol_type': 62
    }
    if os.path.exists(args.output) and os.path.isdir(args.output):
        savename = os.path.join(
            args.output, os.path.split(args.input)[-1] + f"-{all_targets[args.target]}.npy"
        )
        np.save(savename, distribute_dict)
    return distribute_dict


def parse_args(args):
    parser = argparse.ArgumentParser(description='Convert SMILES to canonical type')
    parser.add_argument('--input', default='test.smi', type=str)
    parser.add_argument('--output', default='save_folder', type=str,
                        help='Dirname for output')
    parser.add_argument('--target', default='length', type=str)
    parser.add_argument('--num_workers', default=1, type=int)

    args = parser.parse_args()
    return args


def main(args):
    in_file = args.input
    num_workers = args.num_workers
    consider_symbols = [
        'c', 'C', '(', ')', '1', 'O', '=', '2', 'N', 'n', '3', '[', ']', '@',
        'H', 'F', '-', '4', 'S', 'Cl', '/', 's', 'o', '.', 'Br', '5', '+',
        '#', '\\', '6', 'I', 'P', 'Si', '7', '8', 'B', '%', 'Na', '9', '0',
        'K', 'Sn', 'Se', 'Li', 'Zn', 'Al', 'b', 'As', 'Mg', 'p', 'Ca', 'se',
        'Ag', 'Te', 'Ba', 'Bi', 'Rb', 'Cs', 'Sr', 'te', 'Be'
    ]
    ta = target_analysis(args.target)
    if args.target == 'length':
        specific_func = ta.length_analysis
    elif args.target == 'symbol_type':
        specific_func = ta.symbol_type_analysis
    elif args.target in consider_symbols:
        specific_func = ta.symbol_analysis
    else:
        print(f'Wrong target: {args.target}')

    t1 = time.time()
    combined_result = parallel_operation(in_file, num_workers, specific_func)
    rerange_result_dict(combined_result, consider_symbols, args)
    print(f'| Running time: {time.time() - t1:.3f} s')


def cli_main():
    args = parse_args(sys.argv[1:])
    print(args)
    main(args)


if __name__ == "__main__":
    cli_main()
    print('End!')
