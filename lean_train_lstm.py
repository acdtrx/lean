import torchtext.vocab as vocab
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
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
tb_writer_train, tb_writer_test = lu.setup_tensorboard( training_label )
lu.output_hparams( training_label, net_params, trainer_params, gen_params, lean_vocab )

network = LeanModel( lean_vocab , net_params ).to( device )

trainer = Trainer( device , network , train_data , test_data , trainer_params )

for epoch_no in range( trainer_params['epochs'] ):
    epoch_data = trainer.train_epoch( epoch_no )
    tb_writer_train.add_scalar( 'Loss' , epoch_data[0] , epoch_no )
    tb_writer_train.add_scalar( 'Accuracy' , epoch_data[1] , epoch_no )

    epoch_data = trainer.test_epoch( epoch_no )
    tb_writer_test.add_scalar( 'Loss' , epoch_data[0] , epoch_no )
    tb_writer_test.add_scalar( 'Accuracy' , epoch_data[1] , epoch_no )

    lu.save_network( network, training_label , epoch_no )
