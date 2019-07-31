from termcolor import colored as clr

import lean_utils as lu
import lean_params as lp

label = lu.get_cli_args()

gen_params_test = lp.gen_params_all[label]

training_label = 'baseline-Jul27_16-18-30'
epoch = 4

lean_vocab = lu.load_vocab( gen_params_test['vocab_filename'] )

for epoch_no in range( trainer_params['epochs'] ):
    epoch_samples = lu.load_epoch_samples( epoch_no , training_label )

    for s_input, s_output, s_probs in epoch_samples:
        for token in s_input:
            print( lean_vocab.itos[token] , end="," )
        print()
        for pos, token in enumerate( s_output ):
            if s_probs[pos] > 0.7:
                col = 'green'
            elif s_probs[pos] > 0.3:
                col = 'yellow'
            else:
                col = 'red'
            print( clr( lean_vocab.itos[token] , col ) , end="," )
        print( f'{s_probs.prod() * 100:.2f}' )
