from collections import Counter
from tqdm import tqdm
import pickle
import csv

total_lines = 418236956
v_counter = Counter()

input_filename = './data/auth_users.txt'
counters_filename = './cache/counters_users.pickle'
counters_intermediate_filename = './cache/counters_intermediate_users.pickle'

def process_line( line ):
    for idx, col in enumerate(line):
        if idx == 0:
            continue
        if idx == 1 or idx == 2:
            labels = col.split( '@' )
            v_counter[labels[0]] += 1
            v_counter[labels[1]] += 1
        else:
            v_counter[col] += 1

with open( input_filename , 'r') as f:
    p_bar = tqdm( csv.reader( f ) , total = total_lines , mininterval=1.0)
    for line_no, line in enumerate( p_bar ):
        process_line( line )
        if line_no % 100000 == 0:
            p_bar.set_postfix( v_size = len(v_counter) , refresh=False )
        if line_no % 1000000 == 0:
            with open( counters_intermediate_filename , 'wb' ) as h:
                pickle.dump( v_counter , h )

print( f'Vocabulary size: {len(v_counter)}' )
print( f'Most common: {v_counter.most_common(5)}' )
print( f'Least common: {v_counter.most_common()[:-6:-1]}' )

# v = vocab.Vocab( v_counter , min_freq=40 , specials=['<unk>', '<pad>', '<eos>', '<sos>'] )

with open( counters_filename , 'wb' ) as h:
    pickle.dump( v_counter , h )
    print( f'Dumped to {counters_filename}' )