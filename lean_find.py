import torch
from torch.utils.data import TensorDataset, DataLoader
import torchtext.vocab as vocab

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
    with torch.no_grad():
        for (batch_data,) in test_dl:
            # prepare ground truth
            batch_data = batch_data.to(device)

            hs , _ = network( batch_data[:,:-1] )
            out = network.get_logits( hs )

            batch_items_probs = network.get_probs( out.detach() ).gather( 2 , batch_data[:,1:].unsqueeze(2) ).squeeze(2)
            anomalies += lu.get_anomaly_lines( lean_vocab , batch_data , out , batch_items_probs , device )

    return anomalies


anomaly_lines = find_anomalies( test_data )

print( f'Found {anomaly_lines} anomalies.' )
