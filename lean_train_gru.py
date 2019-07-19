import torchtext.vocab as vocab
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from csvdataset import CSVDataset

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
}

trainer_params = {
    "batch_size": 64,
    "line_size": 10,
    "lr": 0.001
}

input_filename = './cache/tensors_1M.pickle'
input_total_lines = 418236956 # total
input_total_lines = 80000000 # train
input_total_lines = 1000000 # train strip

vocab_filename = './cache/vocab_users.pickle'

with open( vocab_filename , 'rb' ) as h:
    lean_vocab = pickle.load( h )

print(f'Loaded {len(lean_vocab.stoi)} vocab entries.' )

with open( input_filename , 'rb' ) as h:
    input_data = pickle.load( h )

input_data = torch.as_tensor( input_data , dtype=torch.long )

tt_split = round(input_total_lines * 0.8)
train_ds = TensorDataset( input_data[:tt_split] )
test_ds = TensorDataset( input_data[tt_split:] )

train_dl = DataLoader( train_ds , trainer_params['batch_size'] , False )
test_dl = DataLoader( test_ds , trainer_params['batch_size'] , False )

class LeanModel(nn.Module):
    def __init__(self, _vocab, _net_params):
        super().__init__()
        self.vocab_len = len( _vocab.stoi )
        self.net_params = _net_params

        self.embed = nn.Embedding( self.vocab_len , self.net_params['embed_size'] )
        self.rnn = nn.GRU( input_size=self.net_params['embed_size'] , hidden_size=self.net_params['hidden_size'] , batch_first=True )
        self.logits = nn.Linear( in_features = self.net_params['hidden_size'] , out_features = self.vocab_len )

        self.loss = nn.CrossEntropyLoss()
        self.sm = nn.Softmax( dim=2 )

    def forward(self, x ):
        y = self.embed( x )
        return self.rnn( y )

    def get_logits(self, hs):
        return self.logits( hs )

    def get_loss( self, y , gt ):
        return self.loss( y , gt )

    def get_probs( self , y ):
        return self.sm( y )


class Trainer():
    def __init__(self, network, train_dl, test_dl, trainer_params):
        self.network = network
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.trainer_params = trainer_params

        self.optimizer = optim.Adam( self.network.parameters() , lr=self.trainer_params['lr'] )

    def train_epoch(self, epoch_no):
        epoch_loss = 0.0
        epoch_accuracy = 1.0

        # prepare ground truth
        gt_t = torch.full( [ self.trainer_params['batch_size'] , self.trainer_params['line_size'] ] , lean_vocab.stoi['<eos>'] , dtype=torch.long )
        gt_t = gt_t.to(device)

        p_bar = tqdm( self.train_dl , mininterval=1 , leave=True )
        for batch_no, train_data in enumerate(p_bar):
            # prepare ground truth
            train_data = train_data[0].to(device)
            gt_t[:,:-1]=train_data[:,1:]

            hs , _ = self.network( train_data )
            out = self.network.get_logits( hs )

            self.optimizer.zero_grad()
            batch_loss = self.network.get_loss( out.permute(0,2,1) , gt_t )
            batch_loss.backward()
            self.optimizer.step()

            epoch_loss += batch_loss.detach().item()

            batch_probs = self.network.get_probs( out.detach() ).gather( 2 , gt_t.unsqueeze(2) ).squeeze(2)
            batch_accuracy = torch.prod( batch_probs )

            p_bar.set_postfix( epoch_loss=f'{epoch_loss/(batch_no+1):.4f}' , batch_acc = f'{batch_accuracy:.4f}' , refresh=False )

        return epoch_loss / ( batch_no + 1 )


network = LeanModel( lean_vocab , net_params )
network = network.to( device )

trainer = Trainer( network , train_dl , test_dl , trainer_params )

losses = []
for i in range( 1 ):
    epoch_loss = trainer.train_epoch( i )
    losses.append( epoch_loss )

print( losses )