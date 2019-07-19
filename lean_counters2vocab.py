import pickle
import torchtext.vocab as vocab

counters_filename = './cache/counters_users.pickle'
vocab_filename = './cache/vocab_users.pickle'

with open( counters_filename , 'rb' ) as h:
    counters = pickle.load( h )
    print( f'Loaded counters len={len(counters)}' )

lean_vocab = vocab.Vocab( counters, min_freq=40 , specials=['<unk>', '<pad>', '<eos>', '<sos>']  )

with open( vocab_filename , 'wb' ) as h:
    pickle.dump( lean_vocab , h )
    print( f'Saved vocab len={len(lean_vocab.stoi)}' )
