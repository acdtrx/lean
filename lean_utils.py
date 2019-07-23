from torch.utils.tensorboard import SummaryWriter
import pickle
from tabulate import tabulate
import torch

def create_training_label():
    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')

    return current_time

def output_hparams( _training_label, _net_params, _trainer_params, _gen_params, _vocab ):

    output = [
        ['Working Set label', _gen_params['ws_label']],
        ['Working Set size', _gen_params['ws_size']],
        ['Embed size', _net_params['embed_size']],
        ['Hidden size', _net_params['hidden_size']],
        ['Bidirectional LSTM', _net_params['bidirectional']],
        ['Vocab weights alpha', _net_params['weights_alpha']],
        ['Vocabulary <unk> weight', _vocab.freqs['<unk>']],
        ['Vocabulary <eos> weight', _vocab.freqs['<eos>']],
        ['Batch size', _trainer_params['batch_size']],
        ['Learning rate', _trainer_params['lr']]
    ]

    with open( f'./output/{_training_label}.txt' , 'w' ) as h:
        h.write( tabulate( output , ('Parameter' , 'Value') ) )


def setup_tensorboard( _training_label ):
    import os

    log_dir = os.path.join( 'runs', _training_label )
    tb_writer_train = SummaryWriter( f'{log_dir}-train' )
    tb_writer_test = SummaryWriter( f'{log_dir}-test' )

    return tb_writer_train, tb_writer_test

def load_vocab( filename ):
    with open( filename , 'rb' ) as h:
        lean_vocab = pickle.load( h )

    print(f'Loaded {len(lean_vocab.stoi)} vocab entries.' )

    return lean_vocab

def load_data( padding, train_filename, test_filename=None, train_split=0.8 , cut=1.0 ):
    def load_and_prepare_data( filename , padding , cut=1.0 ):
        input_data = torch.load( filename )
        if cut != 1.0:
            input_data = input_data[:round(input_data.size(0) * cut)]
        return torch.cat( [input_data , torch.full( ( input_data.size(0) , 1 ) , padding , dtype=torch.long ) ] , 1 )

    train_data = load_and_prepare_data( train_filename , padding , cut )

    if test_filename != None:
        test_data = load_and_prepare_data( test_filename , padding , cut )
    else:
        data_split = round( train_data.size(0) * train_split )
        train_data, test_data = train_data[:data_split] , train_data[data_split:]

    return train_data, test_data

def save_network(network, training_label, epoch_no):
    torch.save( network.state_dict() , f'./output/model_{training_label}_ep_{epoch_no}.pt' )

def load_network(network, training_label, epoch_no):
    return network.load_state_dict( torch.load( f'./output/model_{training_label}_ep_{epoch_no}.pt' ) )