import torchtext.vocab as vocab
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
import statistics as stats

import pickle
from tqdm import tqdm

# setup device (CPU/GPU)
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

net_params = {
    "embed_size": 128,
    "hidden_size": 512,
    "bidirectional": True
}

trainer_params = {
    "batch_size": 64,
    "line_size": 10,
    "lr": 0.001,
    "epochs": 8
}

# train_filename = './cache/tensors_train.pickle'
train_filename = './cache/tensors_1M.pickle'
# input_total_lines = 418236956 # total


with open( train_filename , 'rb' ) as h:
    input_data = pickle.load( h )

input_data = torch.as_tensor( input_data , dtype=torch.long )

train_ds = TensorDataset( input_data )
test_ds = TensorDataset( input_data[:1] )

train_dl = DataLoader( train_ds , trainer_params['batch_size'] , True )
test_dl = DataLoader( test_ds , trainer_params['batch_size'] , False )

# load vocabulary
vocab_filename = './cache/vocab_users.pickle'

with open( vocab_filename , 'rb' ) as h:
    lean_vocab = pickle.load( h )

lean_vocab.freqs['<unk>'] = 4000
lean_vocab.freqs['<eos>'] = len(input_data)

print(f'Loaded {len(lean_vocab.stoi)} vocab entries.' )

# prepare input_data with <eos> for GT
input_data = torch.cat( [input_data , torch.full( ( input_data.size(0) , 1 ) , lean_vocab.stoi['<eos>'] , dtype=torch.long ) ] , 1 )

class LeanModel(nn.Module):
    def __init__(self, _vocab, _net_params, _trainer_params):
        super().__init__()
        self.model_filename='./output/model.pt'
        self.vocab = _vocab
        self.vocab_len = len(_vocab.stoi)
        self.net_params = _net_params
        self.trainer_params = _trainer_params
        self.dirs = ( 2 if self.net_params['bidirectional'] else 1 )

        self.embed = nn.Embedding( self.vocab_len , self.net_params['embed_size'] )
        self.rnn = nn.LSTM(
            input_size=self.net_params['embed_size'],
            hidden_size=self.net_params['hidden_size'],
            batch_first=True,
            bidirectional=self.net_params['bidirectional']
        )
        self.logits = nn.Linear(
            in_features = self.net_params['hidden_size'] * self.dirs,
            out_features = self.vocab_len
        )

        self.make_weights()

        self.loss = nn.CrossEntropyLoss( weight=self.vocab_weights )
        self.sm = nn.Softmax( dim=2 )
        self.h0 = nn.Parameter( torch.randn( ( self.dirs , self.trainer_params['batch_size'] , self.net_params['hidden_size'] ) ) )
        self.c0 = nn.Parameter( torch.randn( ( self.dirs , self.trainer_params['batch_size'] , self.net_params['hidden_size'] ) ) )

    def make_weights( self ):
        self.vocab_weights = torch.zeros( self.vocab_len , dtype=torch.float )

        alpha = 1000
        for i in range(self.vocab_len):
            self.vocab_weights[ i ] = 1 / ( self.vocab.freqs[self.vocab.itos[i]] + alpha )

    def forward(self, x ):
        y = self.embed( x )
        return self.rnn( y , ( self.h0, self.c0 ) )

    def get_logits(self, hs):
        return self.logits( hs )

    def get_loss( self, y , gt ):
        return self.loss( y , gt )

    def get_probs( self , y ):
        return self.sm( y )

    def save(self):
        torch.save( self.state_dict() , self.model_filename )

    def load(self):
        self.load_state_dict( torch.load( self.model_filename ) )


class Trainer():
    def __init__(self, network, train_dl, test_dl, trainer_params):
        self.network = network
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.trainer_params = trainer_params

        self.optimizer = optim.Adam( self.network.parameters() , lr=self.trainer_params['lr'] )
        self.scheduler = lr_scheduler.ReduceLROnPlateau( self.optimizer , mode='min' , factor=0.1 , threshold=0.1 , patience=1 , verbose=True )

    def train_epoch(self, epoch_no):
        epoch_loss = 0.0
        epoch_accuracy = 0.0

        p_bar = tqdm( self.train_dl , desc=f'Train ep {epoch_no}' , mininterval=1 , leave=True , dynamic_ncols=True )
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
                batch_probs = self.network.get_probs( out.detach() / 1.0 ).gather( 2 , train_data[:,1:].unsqueeze(2) ).squeeze(2)
                batch_accuracy = batch_probs.prod( dim=1 ).mean()
                epoch_accuracy += batch_accuracy

                p_bar.set_postfix(
                    refresh=False,
                    ep_loss=f'{epoch_loss/(batch_no+1):.4f}',
                    btc_acc = f'{batch_accuracy:.4f}',
                    ep_acc=f'{epoch_accuracy/(batch_no+1):.2f}'
                )
        
        self.scheduler.step( epoch_loss )

        return epoch_loss / ( batch_no + 1 ) , epoch_accuracy/( batch_no + 1 )


network = LeanModel( lean_vocab , net_params , trainer_params )
network = network.to( device )

trainer = Trainer( network , train_dl , test_dl , trainer_params )

losses = []
for i in range( trainer_params['epochs'] ):
    epoch_loss = trainer.train_epoch( i )
    losses.append( epoch_loss )

network.save()

for epoch, (loss, acc) in enumerate( losses ):
    print( f'Epoch {epoch} loss: {loss:.4f} accuracy: {acc:.4f}' )