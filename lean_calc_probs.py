import torch
import torchtext.vocab as vocab

from tqdm import tqdm, trange
import matplotlib.pyplot as plt

import lean_utils as lu
from lean_network import LeanModel, LeanNetRunner
import lean_params as lp

gen_params_train = lp.gen_params_all['day7']
gen_params_test = lp.gen_params_all['day8']

training_label = 'baseline-Jul27_16-18-30'
training_epoch = 8
test_filename = f'./cache/tensors_{gen_params_test["ws_label"]}.pt'
probs_filename = f'./cache/probs_{gen_params_test["ws_label"]}.pt'

# setup device (CPU/GPU)
device = lu.get_device()

vocab_filename = f'./cache/vocab_users_{gen_params_train["ws_label"]}.pickle'
lean_vocab = lu.load_vocab( vocab_filename )

network = LeanModel( lean_vocab , lp.net_params )
lu.load_network( network , training_label , training_epoch )
network = network.to( device )

test_data = torch.load( test_filename )

def calculate_probs( network , in_data ):
    probs_out = torch.zeros( in_data.size(0) )
    batch_size = 128
    with torch.no_grad():
        p_bar = tqdm(
            LeanNetRunner( network , in_data , batch_size ),
            desc=f'Get Probs'
        )
        for batch_no, (batch_data, batch_out) in enumerate( p_bar ):
            batch_probs = network.get_probs( batch_out.detach() ).gather( 2 , batch_data[:,1:].unsqueeze(2) ).squeeze(2).prod( dim=1 )
            probs_out[batch_no*batch_size:(batch_no+1)*batch_size] = batch_probs

    return probs_out

probs = calculate_probs( network , test_data )
torch.save( probs , probs_filename )
