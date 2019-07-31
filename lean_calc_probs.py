import torch
import torchtext.vocab as vocab

from tqdm import tqdm, trange
import matplotlib.pyplot as plt

import lean_utils as lu
from lean_network import LeanModel, LeanNetRunner
import lean_params as lp

training_sets_bi = [
    ('baseline-Jul26_10-06-31' , 8 , 'Bi-LSTM 1 day' , 'day8' , True),
    ('baseline0-7-Jul28_14-36-21' , 8 , 'Bi-LSTM 8 days' , 'day0-7-8' , True)
]
training_sets_uni = [
    ('unidir7-Jul29_11-23-32' , 8 , 'LSTM 1 day' , 'day8' , False),
    ('unidir7-Jul30_08-02-11' , 8 , 'LSTM 8 days' , 'day0-7-8' , False)
]

# setup device (CPU/GPU)
device = lu.get_device()

def calculate_probs( network , in_data , training_label ):
    probs_out = torch.zeros( in_data.size(0) )
    batch_size = 128
    with torch.no_grad():
        p_bar = tqdm(
            LeanNetRunner( network , in_data , batch_size ),
            desc=f'Get Probs {training_label}'
        )
        for batch_no, (batch_data, batch_out) in enumerate( p_bar ):
            batch_probs = network.get_probs( batch_out.detach() ).gather( 2 , batch_data[:,1:].unsqueeze(2) ).squeeze(2).prod( dim=1 )
            probs_out[batch_no*batch_size:(batch_no+1)*batch_size] = batch_probs

    return probs_out

def calculate_probs_for_set( training_set ):
    training_label , training_epoch , _ , ws_label , bidirectional = training_set

    vocab_filename = lp.gen_params_all[ws_label]['vocab_filename']
    lean_vocab = lu.load_vocab( vocab_filename )

    test_filename = lp.gen_params_all[ws_label]['tensors_filename']
    probs_filename = f'./cache/probs_{ws_label}_{training_label}_{training_epoch}.pt'

    net_params = lp.net_params
    net_params['bidirectional'] = bidirectional

    network = LeanModel( lean_vocab , net_params )
    lu.load_network( network , training_label , training_epoch )
    network = network.to( device )

    test_data = lu.load_and_prepare_data( test_filename , lean_vocab.stoi['<eos>'] )

    probs = calculate_probs( network , test_data , training_label )
    print(f'Saving {probs_filename}')
    torch.save( probs , probs_filename )

for training_set in training_sets_bi:
    calculate_probs_for_set( training_set )

for training_set in training_sets_uni:
    calculate_probs_for_set( training_set )
