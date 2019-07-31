import torch
import numpy as np

from tqdm import trange
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import lean_utils as lu
import lean_params as lp

label = 'day0-7-8'

gen_params_test = lp.gen_params_all[label]
gen_params_red8 = lp.gen_params_all['redteam8']

training_label = 'baseline0-7-Jul28_14-36-21'
training_epoch = 8

redlabels_filename = f'./cache/redlabels_day8.pt'
probs_filename = f'./cache/probs_{label}_{training_label}_{training_epoch}.pt'

def y_fmt(y, pos):
    decades = [1e9, 1e6, 1e3, 1e0, 1e-3, 1e-6, 1e-9 ]
    suffix  = ["G", "M", "k", "" , "m" , "u", "n"  ]
    if y == 0:
        return str(0)
    for i, d in enumerate(decades):
        if np.abs(y) >=d:
            val = y/float(d)
            signf = len(str(val).split(".")[1])
            if signf == 0:
                return '{val:d} {suffix}'.format(val=int(val), suffix=suffix[i])
            else:
                if signf == 1:
                    if str(val).split(".")[1] == "0":
                       return '{val:d} {suffix}'.format(val=int(round(val)), suffix=suffix[i]) 
                tx = "{"+"val:.{signf}f".format(signf = signf) +"} {suffix}"
                return tx.format(val=val, suffix=suffix[i])

    return y


redlabels = torch.load( redlabels_filename ).data.cpu().numpy()
probs = torch.load( probs_filename ).data.cpu().numpy()
anomm_probs = probs[redlabels == 1]


plt.hist( probs , facecolor='g' )
plt.title( 'Overall probabilities distribution' )
plt.xlabel( 'Probability' )
plt.ylabel( 'Log lines' )
plt.gca().yaxis.set_major_formatter( FuncFormatter( y_fmt ) )
plt.show()
plt.close()

plt.hist( anomm_probs , facecolor='r' )
plt.title( 'Redteam probabilities distribution' )
plt.xlabel( 'Probability' )
plt.ylabel( 'Log lines' )
plt.gca().yaxis.set_major_formatter( FuncFormatter( y_fmt ) )
plt.show()
plt.close()

