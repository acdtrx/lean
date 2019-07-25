import torchtext.vocab as vocab
import torch
import csv

import pickle
from tqdm import tqdm

from lean_params import gen_params

input_filename = './data/auth_users.txt'
vocab_filename = f'./cache/vocab_users_{gen_params["ws_label"]}.pickle'

train_tensors_filename = f'./cache/tensors_train_{gen_params["ws_label"]}.pt'
test_tensors_filename = f'./cache/tensors_test_{gen_params["ws_label"]}.pt'

line_size = 10
create_train = True
create_test = True

# import vocabulary
with open( vocab_filename , 'rb') as h:
    v = pickle.load( h )

def process_line( line ):
    ret = torch.zeros( line_size )
    c = 0
    for idx, col in enumerate(line):
        if idx == 0:
            continue
        if idx == 1 or idx == 2:
            labels = col.split( '@' )
            ret[c] = v.stoi[labels[0]]
            c+=1
            ret[c] = v.stoi[labels[1]]
            c+=1
        else:
            ret[c] = v.stoi[col]
            c+=1
    return ret

with open( input_filename , 'r') as f:
    if create_train:
        t = torch.zeros( [gen_params['ws_size'] , line_size] , dtype=torch.long )
        p_bar = tqdm( csv.reader( f ) , desc="Train processing" , total = gen_params['ws_size'] , mininterval=1.0)
        for line_no, line in enumerate( p_bar ):
            t[line_no,:] = process_line(line)
            if line_no == gen_params['ws_size'] - 1:
                break

        torch.save( t , train_tensors_filename )
        print( f'Saved {train_tensors_filename}' )

    else:
        p_bar = tqdm( csv.reader( f ) , desc="Train skipping" , total = gen_params['ws_size'] , mininterval=1.0)
        for line_no, line in enumerate( p_bar ):
            if line_no == gen_params['ws_size'] - 1:
                break

    if create_test:
        t = torch.zeros( [gen_params['ws_size'] , line_size] , dtype=torch.long )
        p_bar = tqdm( csv.reader( f ) , desc="Test processing" , total = gen_params['ws_size'] , mininterval=1.0)
        for line_no, line in enumerate( p_bar ):
            t[line_no,:] = process_line(line)
            if line_no == gen_params['ws_size'] - 1:
                break

        torch.save( t , test_tensors_filename )
        print( f'Saved {test_tensors_filename}' )
