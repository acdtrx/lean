import torch

from tqdm import trange
import matplotlib.pyplot as plt

import lean_utils as lu
import lean_params as lp

gen_params_test = lp.gen_params_all['day8']
gen_params_red8 = lp.gen_params_all['redteam8']

redlabels_filename = f'./cache/redlabels_{gen_params_test["ws_label"]}.pt'
probs_filename = f'./cache/probs_{gen_params_test["ws_label"]}.pt'


device = lu.get_device()
redlabels = torch.load( redlabels_filename ).to(device)
probs = torch.load( probs_filename ).to(device)

fps , tps = [], []

total_rl = redlabels.sum()
total_lines = probs.size(0)

for i in trange( 1001 ):
    thr = i / 1000

    lines_below = (probs < thr).type( torch.int8 )

    fp = ( ( lines_below - redlabels ) > 0 ).sum()
    tp = total_rl - ( ( redlabels - lines_below ) > 0 ).sum()

    tps.append( tp.item() / total_rl )
    fps.append( fp.item() / total_lines )

plt.figure()
plt.plot( fps , tps )
plt.xlabel( 'False Positives' )
plt.ylabel( 'True Positives' )
plt.show()
