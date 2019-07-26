import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

import lean_utils as lu
from lean_network import LeanModel
from lean_trainer import Trainer

import lean_params as lp

gen_params = lp.gen_params_all['day7']

# setup device (CPU/GPU)
device = lu.get_device()

# input filenames
vocab_filename = f'./cache/vocab_users_{gen_params["ws_label"]}.pickle'
train_filename = f'./cache/tensors_{gen_params["ws_label"]}.pt'
test_filename = f'./cache/tensors_{lp.gen_params_all["day8"]["ws_label"]}.pt'

# load vocabulary
lean_vocab = lu.load_vocab( vocab_filename )

# load training and test
train_data, test_data = lu.load_data( lean_vocab.stoi['<eos>'] , train_filename , test_filename )

#setup tensorboard & friends
training_label = lu.create_training_label()
print( f'Training label: {training_label}' )
tb_train_writer, tb_test_writer = lu.setup_tensorboard( training_label )
lu.output_hparams( training_label, lp.net_params, lp.trainer_params, gen_params, lean_vocab )

# output vocabulary freqs
def vocab_summary( _vocab , _sw ):
    x, y = zip( *_vocab.freqs.most_common( 10 ) )
    plt.figure( figsize=(16,4) )
    plt.bar( x , y )
    plt.xticks( rotation=315 , rotation_mode="anchor" )
    _sw.add_figure( "Vocab/Most_Common_10" , plt.gcf() )

    x, y = zip( *_vocab.freqs.most_common( 50 )[10:] )
    plt.figure( figsize=(16,4) )
    plt.bar( x , y )
    plt.xticks( rotation=315 , rotation_mode="anchor" )
    _sw.add_figure( "Vocab/Most_Common_Next40" , plt.gcf() )

    lf = _vocab.freqs.most_common()
    lfa = []
    for k,v in lf:
        if v >= 10:
            lfa.append( (k,v) )

    x, y = zip( *lfa[-51:] )
    plt.figure( figsize=(16,4) )
    plt.bar( x , y )
    plt.xticks( rotation=315 )
    _sw.add_figure( "Vocab/Least_Common" , plt.gcf() )

vocab_summary( lean_vocab , tb_train_writer )

network = LeanModel( lean_vocab , lp.net_params ).to( device )

trainer = Trainer( device , network , train_data , test_data , lp.trainer_params )

def run_epoch( epoch_no , tests = None , save_network = False):
    epoch_loss, epoch_acc, test_losses = trainer.train_epoch( epoch_no , tests )
    tb_train_writer.add_scalar( 'Epoch/Loss' , epoch_loss , epoch_no )
    tb_train_writer.add_scalar( 'Epoch/Accuracy' , epoch_acc , epoch_no )

    if tests:
        plt.figure( figsize=( 3 , 3 ) )
        plt.plot( test_losses )
        tb_train_writer.add_figure( 'Epoch/Train-Test' , plt.gcf() , epoch_no )

    epoch_loss, epoch_acc = trainer.test_epoch( epoch_no )
    tb_test_writer.add_scalar( 'Epoch/Loss' , epoch_loss , epoch_no )
    tb_test_writer.add_scalar( 'Epoch/Accuracy' , epoch_acc , epoch_no )

    if save_network:
        lu.save_network( network, training_label , epoch_no )

for epoch_no in range(0, lp.trainer_params['epochs'] ):
    run_epoch( epoch_no , save_network=True )
