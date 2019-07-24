import torch
from torch.utils.data import TensorDataset, DataLoader
import torchtext.vocab as vocab

from termcolor import colored as clr

import lean_utils as lu
from lean_network import LeanModel

from tqdm import tqdm

from lean_params import net_params, gen_params

training_label = 'Jul24_14-34-58'
test_filename = f'./cache/tensors_train_{gen_params["ws_label"]}.pt'

# setup device (CPU/GPU)
device = lu.get_device()

vocab_filename = f'./cache/vocab_users_{gen_params["ws_label"]}.pickle'
lean_vocab = lu.load_vocab( vocab_filename )

network = LeanModel( lean_vocab , net_params )
lu.load_network( network , training_label , 3 )
network = network.to( device )

_ , test_data = lu.load_data( lean_vocab.stoi['<eos>'] , test_filename , train_split=0.0 )

def find_anomalies( data ):
    test_ds = TensorDataset( data )
    test_dl = DataLoader( test_ds , 64 , False )
    anomalies = 0
    anomaly_lines = []
    with torch.no_grad():
        p_bar = tqdm( test_dl , desc=f'Find' , mininterval=1 , leave=True , dynamic_ncols=True )
        for batch_no, (batch_data,) in enumerate(p_bar):
            # prepare ground truth
            batch_data = batch_data.to(device)

            hs , _ = network( batch_data[:,:-1] )
            out = network.get_logits( hs )

            batch_items_probs = network.get_probs( out.detach() ).gather( 2 , batch_data[:,1:].unsqueeze(2) ).squeeze(2)
            batch_anomaly_lines = lu.get_anomaly_lines( lean_vocab , batch_data , out , batch_items_probs , device )
            anomaly_lines += batch_anomaly_lines

            # batch_lines_probs = batch_items_probs.prod( dim=1 )

            # anomalies += batch_lines_probs[ batch_lines_probs < 0.05 ].size(0)
            # p_bar.set_postfix( anomalies=anomalies, refresh=False )
    return anomaly_lines


anomaly_lines = find_anomalies( test_data )

for s_input, s_output, s_probs in anomaly_lines:
    for token in s_input:
        print( lean_vocab.itos[token] , end="," )
    print()
    for pos, token in enumerate( s_output ):
        if s_probs[pos] > 0.7:
            col = 'green'
        elif s_probs[pos] > 0.3:
            col = 'yellow'
        else:
            col = 'red'
        print( clr( lean_vocab.itos[token] , col ) , end="," )
    print( f'{s_probs.prod() * 100:.2f}' )