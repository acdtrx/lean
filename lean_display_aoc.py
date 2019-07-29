import torch
from sklearn import metrics

from tqdm import trange
import matplotlib.pyplot as plt

import lean_utils as lu
import lean_params as lp

label = 'day0-7-8'
training_label = 'baseline0-7-Jul28_14-36-21'
training_epoch = 4

redlabels_filename = f'./cache/redlabels_day8.pt'
probs_filename = f'./cache/probs_{label}_{training_label}_{training_epoch}.pt'

device = lu.get_device()
redlabels = torch.load( redlabels_filename ).to(device)
probs = torch.load( probs_filename ).to(device)

fpr, tpr, thresholds = metrics.roc_curve(redlabels.data.cpu().numpy() , probs.data.cpu().numpy() , pos_label=0 )
print( metrics.auc(fpr, tpr) )

plt.figure()
plt.plot( fpr , tpr )
plt.xlabel( 'False Positives' )
plt.ylabel( 'True Positives' )
plt.show()
plt.close()