import torch
from sklearn import metrics

from tqdm import trange
import matplotlib.pyplot as plt

import lean_utils as lu
import lean_params as lp

training_sets_bi = [
    ('baseline-Jul26_10-06-31' , 8 , 'Bi-LSTM 1 day' , 'day8' , True),
    ('baseline0-7-Jul28_14-36-21' , 8 , 'Bi-LSTM 8 days' , 'day0-7-8' , True)
]
training_sets_uni = [
    ('unidir7-Jul29_11-23-32' , 8 , 'LSTM 1 day' , 'day8' , False),
    ('unidir7-Jul30_08-02-11' , 8 , 'LSTM 8 days' , 'day0-7-8' , False)
]

def plot_for_set( training_set ):
    training_label , training_epoch , legend_label , ws_label , bidirectional = training_set
    redlabels_filename = f'./cache/redlabels_day8.pt'
    probs_filename = f'./cache/probs_{ws_label}_{training_label}_{training_epoch}.pt'

    redlabels = torch.load( redlabels_filename )
    probs = torch.load( probs_filename )

    fpr, tpr, thresholds = metrics.roc_curve(redlabels.data.cpu().numpy() , 1-probs.data.cpu().numpy() , pos_label=1 )
    auc = metrics.auc(fpr, tpr)

    plt.plot( fpr , tpr , label=f'{legend_label} [{auc:.4f}]' )

def prg_for_set( training_set ):
    training_label , training_epoch , legend_label , ws_label , bidirectional = training_set
    redlabels_filename = f'./cache/redlabels_day8.pt'
    probs_filename = f'./cache/probs_{ws_label}_{training_label}_{training_epoch}.pt'

    redlabels = torch.load( redlabels_filename ).data.cpu().numpy()
    probs = torch.load( probs_filename ).data.cpu().numpy()

    precision, recall, thresholds = metrics.precision_recall_curve(redlabels , 1-probs , pos_label=1 )

    average_precision = metrics.average_precision_score(redlabels, 1-probs , pos_label=1)

    plt.figure()
    plt.plot(recall, precision, marker='o')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'{legend_label} Precision-Recall curve: AP={average_precision:.2f}' )

    plt.show()
    plt.close()

plt.figure()
for training_set in training_sets_bi:
    plot_for_set( training_set )

for training_set in training_sets_uni:
    plot_for_set( training_set )

plt.title( 'RoC and AuC' )
plt.xlabel( 'False Positives' )
plt.ylabel( 'True Positives' )
plt.legend( loc="lower right" )

plt.savefig( './output/auc.png' , facecolor='w'  )
plt.show()
plt.close()
