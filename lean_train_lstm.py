import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

import lean_utils as lu
from lean_network import LeanModel
from lean_trainer import Trainer

from lean_params import net_params, trainer_params, gen_params

# setup device (CPU/GPU)
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

vocab_filename = f'./cache/vocab_users_{gen_params["ws_label"]}.pickle'
train_filename = f'./cache/tensors_train_{gen_params["ws_label"]}.pt'

# load vocabulary
lean_vocab = lu.load_vocab( vocab_filename )

# load training and test
train_data, test_data = lu.load_data( lean_vocab.stoi['<eos>'] , train_filename , cut=0.5 )

#setup tensorboard & friends
training_label = lu.create_training_label()
print( f'Training label: {training_label}' )
tb_train_writer, tb_test_writer = lu.setup_tensorboard( training_label )
lu.output_hparams( training_label, net_params, trainer_params, gen_params, lean_vocab )

# output vocabulary freqs
def vocab_summary( _vocab , _sw ):
    x, y = zip( *_vocab.freqs.most_common( 50 ) )
    plt.figure( figsize=(9,3) )
    plt.bar( x , y )
    _sw.add_figure( "Vocab most common" , plt.gcf() )

    x, y = zip( *_vocab.freqs.most_common( )[-51:] )
    plt.figure( figsize=(9,3) )
    plt.bar( x , y )
    _sw.add_figure( "Vocab least common" , plt.gcf() )

vocab_summary( lean_vocab , tb_train_writer )

network = LeanModel( lean_vocab , net_params ).to( device )

trainer = Trainer( device , network , train_data , test_data , trainer_params )

def run_epoch( epoch_no , tests = None , save_network = False):
    epoch_loss, epoch_acc, test_losses = trainer.train_epoch( epoch_no , tests )
    tb_train_writer.add_scalar( 'Loss' , epoch_loss , epoch_no )
    tb_train_writer.add_scalar( 'Accuracy' , epoch_acc , epoch_no )

    if tests:
        plt.figure( figsize=( 3 , 3 ) )
        plt.plot( test_losses )
        tb_train_writer.add_figure( 'Train-Tests loss' , plt.gcf() , epoch_no )

    epoch_loss, epoch_acc = trainer.test_epoch( epoch_no )
    tb_test_writer.add_scalar( 'Loss' , epoch_loss , epoch_no )
    tb_test_writer.add_scalar( 'Accuracy' , epoch_acc , epoch_no )

    if save_network:
        lu.save_network( network, training_label , epoch_no )

run_epoch(0 , 20)

for epoch_no in range(1, trainer_params['epochs'] ):
    run_epoch( epoch_no )
