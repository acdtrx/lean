import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader

def get_device( force_cpu = False):
    if not force_cpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    return device

def create_training_label( _label=None ):
    from datetime import datetime
    label = datetime.now().strftime('%b%d_%H-%M-%S')

    if _label:
        label = f'{_label}-{label}'

    return label

def output_hparams( _sw , _training_label, _net_params, _trainer_params, _gen_params, _vocab ):
    def md_table( _params , _titles = None ):
        if not _titles:
            _titles = ['Name' , 'Value']

        out_table = "|".join( _titles ) + '\n'
        out_table += ":---|---:" + '\n'

        for line in _params:
            out_table += ( line[0] + '|' + str( line[1] ) ) + '\n'

        return out_table

    _sw.add_text( 'HyperParameters/Network' , md_table( [
        ['Embed size', _net_params['embed_size']],
        ['Hidden size', _net_params['hidden_size']],
        ['Bidirectional LSTM', _net_params['bidirectional']],
        ['Vocab weights alpha', _net_params['weights_alpha']]
    ] ) )

    _sw.add_text( 'HyperParameters/Trainer' , md_table( [
        ['Batch size', _trainer_params['batch_size']],
        ['Learning rate', _trainer_params['lr']],
        ['Computer accuracy every', _trainer_params['compute_acc_every']]
    ] ) )

    _sw.add_text( 'HyperParameters/Others' , md_table( [
        ['Working Set label', _gen_params['ws_label']],
        ['Vocabulary <unk> weight', _vocab.freqs['<unk>']],
        ['Vocabulary <eos> weight', _vocab.freqs['<eos>']]
    ] ) )

def setup_tensorboard( _training_label ):
    import os
    from torch.utils.tensorboard import SummaryWriter

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

def get_anomaly_lines( _vocab , _input , _output , _probs , _device ):
    from termcolor import colored as clr

    def get_item( _i , _o , _p ):
        if _i != _o:
            el = f'{_vocab.itos[_i]}/{_vocab.itos[_o]}'
        else:
            el = _vocab.itos[_i]

        if _p > 0.75:
            p_color = None
        elif _p > 0.5:
            p_color = 'blue'
        elif _p > 0.25:
            p_color = 'yellow'
        else:
            p_color = 'red'

        return clr( el , p_color )

    b_anomalies = 0
    for line_no, (i_line, o_line, p_line) in enumerate( zip( _input , _output , _probs ) ):
        o_line = torch.cat( [ _input[line_no:line_no+1, 0] , o_line.argmax(dim=1)] )
        p_line = torch.cat( [ torch.as_tensor( [1.0] ).to(_device) , p_line ] )
        if ( i_line != o_line ).sum() != 0:
            b_anomalies += 1
            l_elems = []
            l_elems.append( f'{get_item(i_line[0], o_line[0] , p_line[0])}@{get_item(i_line[1], o_line[1] , p_line[1])}' ) #src_user
            l_elems.append( f'{get_item(i_line[2], o_line[2] , p_line[2])}@{get_item(i_line[3], o_line[3] , p_line[3])}' ) #dst_user
            for i in range(4,11):
                l_elems.append( get_item( i_line[i] , o_line[i] , p_line[i] ) )
            l_elems.append( clr( f'{p_line.prod() * 100:.4f}%' , attrs=['bold'] ) )
            print( ",".join( l_elems ) )

    return b_anomalies

def csv_parse_by_time( _filename, _start_time, _end_time ):
    import csv
    from tqdm import tqdm

    with open( _filename , 'r') as csv_file:
        csv_reader = csv.reader( csv_file )
        p_bar = tqdm( desc = "Skipping" , total = _start_time )
        last_time = "0"
        for line in csv_reader:
            if int(line[0]) >= _start_time:
                p_bar.update( 1 )
                break
            if last_time != line[0]:
                last_time = line[0]
                p_bar.update( 1 )
        p_bar.close()

        #output first line
        yield line

        p_bar = tqdm( desc = "Processing" , total=_end_time - _start_time )
        last_line = "0"
        for line_no , line in enumerate( csv_reader ):
            if int(line[0]) >= _end_time:
                p_bar.update( 1 )
                break
            if line[0] != last_line:
                last_line = line[0]
                p_bar.update( 1 )
            yield line
        p_bar.close()
