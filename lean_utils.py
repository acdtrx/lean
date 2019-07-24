from torch.utils.tensorboard import SummaryWriter
import pickle
from tabulate import tabulate
import torch
import random

def get_device( force_cpu = False):
    if not force_cpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    return device

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
        ['Learning rate', _trainer_params['lr']],
        ['Computer accuracy every', _trainer_params['compute_acc_every']]
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

def save_epoch_samples( epoch_no , training_label , epoch_samples ):
    with open( f'./output/samples_{training_label}_{epoch_no}.pickle' , 'wb' ) as h:
        pickle.dump( epoch_samples , h )

def load_epoch_samples( epoch_no , training_label ):
    with open( f'./output/samples_{training_label}_{epoch_no}.pickle' , 'rb' ) as h:
        epoch_samples = pickle.load( h )

    return epoch_samples

def sample_log_line( _vocab , _input , _output , _probs , _device ):
    idx = random.randint( 0, len(_input) - 1 )
    return (
        _input[idx],
        torch.cat( [ _input[idx:idx+1,0] , _output[idx].argmax(dim=1)] ),
        torch.cat( [ torch.as_tensor( [1.0] ).to(_device) , _probs[idx] ] )
    )

def get_anomaly_lines( _vocab , _input , _output , _probs , _device ):
    al = []
    for line_no, line in enumerate(_input):
        if _probs[line_no].min() < 0.7:
            al.append(
                (
                    line,
                    torch.cat( [ _input[line_no:line_no+1, 0] , _output[line_no].argmax(dim=1)] ),
                    torch.cat( [ torch.as_tensor( [1.0] ).to(_device) , _probs[line_no] ] )
                )
            )
    return al