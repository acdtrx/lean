from torch.utils.tensorboard import SummaryWriter
import pickle

def setup_tensorboard(_net_params, _trainer_params, _vocab):
    import os
    from datetime import datetime

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join( 'runs', current_time )
    tb_writer_train = SummaryWriter( f'{log_dir}-train' )
    tb_writer_test = SummaryWriter( f'{log_dir}-test' )

    tb_writer_train.add_text( 'Embed size' , str(_net_params['embed_size']) )
    tb_writer_train.add_text( 'Hidden size' , str(_net_params['hidden_size']) )
    tb_writer_train.add_text( 'Bidirectional LSTM' , str(_net_params['bidirectional']) )
    tb_writer_train.add_text( 'Vocab weights alpha' , str(_net_params['weights_alpha']) )

    tb_writer_train.add_text( 'Batch size' , str(_trainer_params['batch_size']) )
    tb_writer_train.add_text( 'Learning rate' , str(_trainer_params['lr']) )

    tb_writer_train.add_text( 'Vocabulary <unk> weight' , str(_vocab.freqs['<unk>']) )
    tb_writer_train.add_text( 'Vocabulary <eos> weight' , str(_vocab.freqs['<eos>']) )

    return tb_writer_train, tb_writer_test

def load_vocab( eos_freq ):
    vocab_filename = './cache/vocab_users.pickle'

    with open( vocab_filename , 'rb' ) as h:
        lean_vocab = pickle.load( h )

    lean_vocab.freqs['<unk>'] = 50000
    lean_vocab.freqs['<eos>'] = eos_freq

    print(f'Loaded {len(lean_vocab.stoi)} vocab entries.' )

    return lean_vocab
