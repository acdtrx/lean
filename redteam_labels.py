import torch

from tqdm import tqdm

import lean_utils as lu
import lean_params as lp

gen_params_test = lp.gen_params_all['day8']
gen_params_red8 = lp.gen_params_all['redteam8']

redteam_filename = f'./cache/tensors_{gen_params_red8["ws_label"]}.pt'
test_filename = f'./cache/tensors_{gen_params_test["ws_label"]}.pt'
redlabels_filename = f'./cache/redlabels_{gen_params_test["ws_label"]}.pt'

device = lu.get_device()
redteam_data = torch.load( redteam_filename ).to( device )
test_data = torch.load( test_filename ).to( device )


label_data = torch.zeros( test_data.size(0) , dtype=torch.int8 ).to( device )
line_len = test_data.size(1)
missing = []
for rt_line in tqdm( redteam_data ):
    rt_truth = ( ( test_data == rt_line ).sum( 1 ) == line_len)
    rt_pos = (rt_truth == 1).nonzero().squeeze()
    label_data[ rt_pos ] = 1

print( f'Found {label_data.sum()} matches.' )
print( f"Saving labels to {redlabels_filename}" )
torch.save( label_data , redlabels_filename )
