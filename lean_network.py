import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class LeanModel(nn.Module):
    def __init__(self, _vocab, _net_params):
        super().__init__()
        self.vocab = _vocab
        self.vocab_len = len(_vocab.stoi)
        self.net_params = _net_params
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

    def make_weights( self ):
        self.vocab_weights = torch.zeros( self.vocab_len , dtype=torch.float )

        for i in range(self.vocab_len):
            self.vocab_weights[ i ] = 1 / ( self.vocab.freqs[self.vocab.itos[i]] + self.net_params['weights_alpha'] )

    def forward(self, x ):
        y = self.embed( x )
        return self.rnn( y )

    def get_logits(self, hs):
        return self.logits( hs )

    def get_loss( self, y , gt ):
        return self.loss( y , gt )

    def get_probs( self , y ):
        return self.sm( y )


class LeanNetRunner():
    def __init__(self, _network, _input_data , _batch_size):
        self.network = _network
        self.input_data = _input_data
        self.batch_size = _batch_size
        self.dl_iter = None

        self.device = next(_network.parameters()).device

        self.data_loader = DataLoader( TensorDataset( _input_data ) , _batch_size , False )

    def __iter__(self):
        self.dl_iter = iter(self.data_loader)
        return self

    def __next__(self):
        (batch_data,) = next(self.dl_iter)
        batch_data = batch_data.to( self.device )

        hs , _ = self.network( batch_data[:,:-1] )
        batch_out = self.network.get_logits( hs )

        return batch_data, batch_out

    def __len__(self):
        return len(self.data_loader)
