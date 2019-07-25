import pickle
import torchtext.vocab as vocab
from lean_params import gen_params

counters_filename = f'./cache/counters_users_{gen_params["ws_label"]}.pickle'
vocab_filename = f'./cache/vocab_users_{gen_params["ws_label"]}.pickle'

with open( counters_filename , 'rb' ) as h:
    counters = pickle.load( h )
    print( f'Loaded counters len={len(counters)}' )

lean_vocab = vocab.Vocab( counters, min_freq=gen_params['vocab_cutoff'] , specials=['<unk>', '<pad>']  )

with open( vocab_filename , 'wb' ) as h:
    pickle.dump( lean_vocab , h )
    print( f'Saved {vocab_filename} len={len(lean_vocab.stoi)}' )
