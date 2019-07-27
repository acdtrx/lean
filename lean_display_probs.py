import torch

from tqdm import tqdm, trange
import matplotlib.pyplot as plt

import lean_utils as lu
from lean_network import LeanModel, LeanNetRunner
import lean_params as lp

probs_filename = f'./cache/probs_{lp.gen_params_test["ws_label"]}.pt'

def load_probs( filename ):
    return torch.load( filename )

def plot_elems_under_thr( probs ):
    x , y = [] , []
    for i in trange( 101 ):
        x.append( i )
        y.append( ( probs < i ).sum().item() * 100 / probs.size(0) )

    plt.figure()
    plt.plot( x , y )
    plt.show()
    plt.close()

    x , y = [] , []
    for i in trange( 101 ):
        x.append( i / 100 )
        y.append( ( probs < (i / 100) ).sum().item() * 100 / probs.size(0) )

    plt.figure()
    plt.plot( x , y )
    plt.show()
    plt.close()

probs = torch.load( probs_filename )
plot_elems_under_thr( probs )
