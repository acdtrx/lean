import pandas as pd
import torchtext.vocab as vocab
import torch
import csv

import pickle
from tqdm import tqdm

# total_lines = 418236956
train_lines = 80000000 # 80M
test_lines = 80000000 #80M

input_filename = './data/auth_users.txt'
vocab_filename = './cache/vocab_users.pickle'

line_size = 10
create_train = False
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
        t = torch.zeros( [train_lines , line_size] , dtype=torch.int32 )
        p_bar = tqdm( csv.reader( f ) , desc="Train processing" , total = train_lines , mininterval=1.0)
        for line_no, line in enumerate( p_bar ):
            t[line_no,:] = process_line(line)
            if line_no == train_lines - 1:
                break

        with open( './cache/tensors_train.pickle' , 'wb') as h:
            pickle.dump( t , h )

    else:
        p_bar = tqdm( csv.reader( f ) , desc="Train skipping" , total = train_lines , mininterval=1.0)
        for line_no, line in enumerate( p_bar ):
            if line_no == train_lines - 1:
                break

    if create_test:
        t = torch.zeros( [test_lines , line_size] , dtype=torch.int32 )
        p_bar = tqdm( csv.reader( f ) , desc="Test processing" , total = test_lines , mininterval=1.0)
        for line_no, line in enumerate( p_bar ):
            t[line_no,:] = process_line(line)
            if line_no == test_lines - 1:
                break

        with open( './cache/tensors_test.pickle' , 'wb') as h:
            pickle.dump( t , h )