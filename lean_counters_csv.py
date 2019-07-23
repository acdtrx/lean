from collections import Counter
from tqdm import tqdm
import pickle
import csv
from lean_params import gen_params

v_counter = Counter()

input_filename = './data/auth_users.txt'
counters_filename = f'./cache/counters_users_{gen_params["ws_label"]}.pickle'

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
    p_bar = tqdm( csv.reader( f ) , total = gen_params['ws_size'] )
    for line_no, line in enumerate( p_bar ):
        if line_no == gen_params['ws_size']:
            break
        process_line( line )
        if line_no % 100000 == 0:
            p_bar.set_postfix( v_size = len(v_counter) , refresh=False )

v_counter['<eos>'] = gen_params['ws_size']

print( f'Vocabulary size: {len(v_counter)}' )
print( f'Most common: {v_counter.most_common(5)}' )
print( f'Least common: {v_counter.most_common()[:-6:-1]}' )

with open( counters_filename , 'wb' ) as h:
    pickle.dump( v_counter , h )
    print( f'Dumped to {counters_filename}' )