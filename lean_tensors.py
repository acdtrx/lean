import torchtext.vocab as vocab
import torch
import csv

import pickle
from tqdm import tqdm

import lean_utils as lu
import lean_params as lp

gen_params = lp.gen_params_all['day0-7']

input_filename = gen_params['csv_filename']
vocab_filename = f'./cache/vocab_users_{lp.gen_params_all["day0-7"]["ws_label"]}.pickle'

tensors_filename = f'./cache/tensors_{gen_params["ws_label"]}.pt'

line_size = 10

# import vocabulary
v = lu.load_vocab( vocab_filename )

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

tbs = 10000
tbi = 0
t = torch.zeros( [ tbs , line_size] , dtype=torch.long )
tensors_arr = []
for line in lu.csv_parse_by_time( input_filename , gen_params['ws_start_time'] , gen_params['ws_end_time'] ):
    if tbi == tbs:
        tensors_arr.append( t )
        tbi = 0
        t = torch.zeros( [ tbs , line_size] , dtype=torch.long )

    t[tbi,:] = process_line(line)
    tbi += 1
tensors_arr.append( t[:tbi] )

out_t = torch.cat( tensors_arr , dim=0 )

torch.save( out_t , tensors_filename )
print( f'Saved {tensors_filename} Size {out_t.size(0)}' )
