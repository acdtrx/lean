import torch
import torch.nn as nn

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

