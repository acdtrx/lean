import torchtext.vocab as vocab
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import statistics as stats

from torch.utils.tensorboard import SummaryWriter
import lean_utils as lu

from tqdm import tqdm

from lean_network import LeanModel

from lean_params import net_params, trainer_params

# setup device (CPU/GPU)
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# train_filename = './cache/tensors_train.pt'
train_filename = './cache/tensors_1M.pt'
# input_total_lines = 418236956 # total

# load vocabulary
lean_vocab = lu.load_vocab( 10000000 )

# load training and test
train_data, test_data = lu.load_data( lean_vocab.stoi['<eos>'] , train_filename , cut=0.2 )

#setup tensorboard & friends
training_label = lu.create_training_label()
tb_writer_train, tb_writer_test = lu.setup_tensorboard( training_label )
lu.output_hparams( training_label, net_params, trainer_params, lean_vocab )

# trainer class
class Trainer():
    def __init__(self, network, train_data, test_data, trainer_params):
        self.network = network
        self.trainer_params = trainer_params

        train_ds = TensorDataset( train_data )
        test_ds = TensorDataset( test_data )

        self.train_dl = DataLoader( train_ds , trainer_params['batch_size'] , False )
        self.test_dl = DataLoader( test_ds , trainer_params['batch_size'] , False )

        self.optimizer = optim.Adam( self.network.parameters() , lr=self.trainer_params['lr'] )
        self.scheduler = lr_scheduler.ReduceLROnPlateau( self.optimizer , mode='min' , factor=0.1 , threshold=0.1 , patience=1 , verbose=True )

    def train_epoch(self, epoch_no):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        epoch_accuracy_p = 0
        display_acc_every = 100

        p_bar = tqdm( self.train_dl , desc=f'Trn {epoch_no}' , mininterval=1 , leave=True , dynamic_ncols=True )
        for batch_no, train_data in enumerate(p_bar):
            # prepare ground truth
            train_data = train_data[0].to(device)

            hs , _ = self.network( train_data[:,:-1] )
            out = self.network.get_logits( hs )

            self.optimizer.zero_grad()
            batch_loss = self.network.get_loss( out.permute(0,2,1) , train_data[:,1:] )
            batch_loss.backward()
            self.optimizer.step()

            epoch_loss += batch_loss.detach().item()

            if batch_no % 100 == 0:
                batch_probs = self.network.get_probs( out.detach() / 0.8 ).gather( 2 , train_data[:,1:].unsqueeze(2) ).squeeze(2)
                batch_accuracy = batch_probs.prod( dim=1 ).mean()
                epoch_accuracy += batch_accuracy
                epoch_accuracy_p = epoch_accuracy*100/(1 + batch_no / display_acc_every )

                p_bar.set_postfix(
                    refresh=False,
                    ep_loss=f'{epoch_loss/(batch_no+1):.4f}',
                    btc_acc = f'{batch_accuracy*100:.2f}%',
                    ep_acc=f'{epoch_accuracy_p:.2f}%'
                )
        
        self.scheduler.step( epoch_loss )

        return epoch_loss / ( batch_no + 1 ) , epoch_accuracy_p

    def test_epoch(self, epoch_no):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        epoch_accuracy_p = 0
        display_acc_every = 100

        with torch.no_grad():
            p_bar = tqdm( self.test_dl , desc=f'Tst {epoch_no}' , mininterval=1 , leave=True , dynamic_ncols=True )
            for batch_no, test_data in enumerate(p_bar):
                # prepare ground truth
                test_data = test_data[0].to(device)

                hs , _ = self.network( test_data[:,:-1] )
                out = self.network.get_logits( hs )

                batch_loss = self.network.get_loss( out.permute(0,2,1) , test_data[:,1:] )
                epoch_loss += batch_loss.detach().item()

                if batch_no % 100 == 0:
                    batch_probs = self.network.get_probs( out.detach() / 0.8 ).gather( 2 , test_data[:,1:].unsqueeze(2) ).squeeze(2)
                    batch_accuracy = batch_probs.prod( dim=1 ).mean()
                    epoch_accuracy += batch_accuracy
                    epoch_accuracy_p = epoch_accuracy*100/(1 + batch_no / display_acc_every )

                    p_bar.set_postfix(
                        refresh=False,
                        ep_loss=f'{epoch_loss/(batch_no+1):.4f}',
                        btc_acc = f'{batch_accuracy*100:.2f}%',
                        ep_acc=f'{epoch_accuracy_p:.2f}%'
                    )

        return epoch_loss / ( batch_no + 1 ) , epoch_accuracy_p

network = LeanModel( lean_vocab , net_params ).to( device )

trainer = Trainer( network , train_data , test_data , trainer_params )

for epoch_no in range( trainer_params['epochs'] ):
    epoch_data = trainer.train_epoch( epoch_no )
    tb_writer_train.add_scalar( 'Loss' , epoch_data[0] , epoch_no )
    tb_writer_train.add_scalar( 'Accuracy' , epoch_data[1] , epoch_no )

    epoch_data = trainer.test_epoch( epoch_no )
    tb_writer_test.add_scalar( 'Loss' , epoch_data[0] , epoch_no )
    tb_writer_test.add_scalar( 'Accuracy' , epoch_data[1] , epoch_no )

    lu.save_network( network, training_label , epoch_no )
