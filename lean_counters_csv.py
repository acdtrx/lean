from collections import Counter
import pickle
import lean_params as lp
import lean_utils as lu

gen_params = lp.gen_params_all['day7']

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

for line_no, line in enumerate( lu.csv_parse_by_time( input_filename , gen_params['ws_start_time'] , gen_params['ws_end_time'] ) ):
    process_line( line )

v_counter['<eos>'] = line_no
v_counter['<sos>'] = line_no
unk_count = 0
for val in v_counter.values():
    if val < gen_params['vocab_cutoff']:
        unk_count += val

v_counter['<unk>'] = unk_count

print( f'Vocabulary size: {len(v_counter)}' )
print( f'<unk> freq: {v_counter["<unk>"]}' )
print( f'<eos> & <sos> freq: {v_counter["<eos>"]}' )
print( f'Most common: {v_counter.most_common(5)}' )
print( f'Least common: {v_counter.most_common()[:-6:-1]}' )

with open( counters_filename , 'wb' ) as h:
    pickle.dump( v_counter , h )
    print( f'Dumped to {counters_filename}' )