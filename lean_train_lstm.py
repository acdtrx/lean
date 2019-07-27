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
vocab_filename = f'./cache/vocab_users_{lp.gen_params_all[lp.vocab_params_label]["ws_label"]}.pickle'
train_filename = f'./cache/tensors_{gen_params["ws_label"]}.pt'
test_filename = f'./cache/tensors_{lp.gen_params_all["day8"]["ws_label"]}.pt'

# load vocabulary
lean_vocab = lu.load_vocab( vocab_filename )

# load training and test
train_data, test_data = lu.load_data( lean_vocab.stoi['<eos>'] , train_filename , test_filename )

#setup tensorboard & friends
training_label = lu.create_training_label('baseline')
# training_label = 'baseline-Jul26_10-06-31'
print( f'Training label: {training_label}' )
tb_train_writer, tb_test_writer = lu.setup_tensorboard( training_label )
lu.output_hparams( tb_train_writer, training_label, lp.net_params, lp.trainer_params, gen_params, lean_vocab )


# output vocabulary freqs
# vocab_summary( lean_vocab , tb_train_writer )

network = LeanModel( lean_vocab , lp.net_params ).to( device )

trainer = Trainer( network , train_data , test_data , lp.trainer_params )

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

if lp.trainer_params['starting_epoch'] != 0:
    lu.load_network( network , training_label , lp.trainer_params['starting_epoch'] - 1 )

for epoch_no in range(lp.trainer_params['starting_epoch'], lp.trainer_params['starting_epoch']+lp.trainer_params['epochs'] ):
    run_epoch( epoch_no , save_network=True )
