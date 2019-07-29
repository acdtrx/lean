import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim

import lean_utils as lu
from tqdm import tqdm
from lean_network import LeanNetRunner

# trainer class
class Trainer():
    def __init__(self, network, train_data, test_data, trainer_params):
        self.network = network
        self.trainer_params = trainer_params

        self.train_data = train_data
        self.test_data = test_data

        self.optimizer = optim.Adam( self.network.parameters() , lr=self.trainer_params['lr'] )
        # self.scheduler = lr_scheduler.ReduceLROnPlateau( self.optimizer , mode='min' , factor=0.1 , threshold=0.1 , patience=1 , verbose=True )
        self.scheduler = lr_scheduler.StepLR( self.optimizer , 3 , 0.1 )

    def train_epoch(self, epoch_no, tests = None):
        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']

        epoch_loss = 0.0
        epoch_accuracy = 0.0
        epoch_accuracy_p = 0

        test_losses = []
        if tests:
            test_every = len( self.train_dl ) // tests

        p_bar = tqdm(
            LeanNetRunner( self.network , self.train_data , self.trainer_params['batch_size'] ),
            desc=f'Trn {epoch_no}'
        )
        for batch_no, (batch_data, batch_out) in enumerate( p_bar ):
            batch_no += 1

            self.optimizer.zero_grad()
            batch_loss = self.network.get_loss( batch_out.permute(0,2,1) , batch_data[:,1:] )
            batch_loss.backward()
            self.optimizer.step()

            epoch_loss += batch_loss.detach().item()

            if batch_no % self.trainer_params['compute_acc_every'] == 0:
                batch_probs = self.network.get_probs( batch_out.detach() ).gather( 2 , batch_data[:,1:].unsqueeze(2) ).squeeze(2)
                batch_accuracy = batch_probs.prod( dim=1 ).mean()
                epoch_accuracy += batch_accuracy
                epoch_accuracy_p = epoch_accuracy*100/(batch_no / self.trainer_params['compute_acc_every'] )

                p_bar.set_postfix(
                    refresh=False,
                    ep_loss=f'{epoch_loss/batch_no:.4f}',
                    btc_acc = f'{batch_accuracy*100:.2f}%',
                    ep_acc=f'{epoch_accuracy_p:.2f}%'
                )

            # if tests and batch_no % test_every == 0:
            #     test_loss, _ = self.test_epoch( epoch_no )
            #     test_losses.append( test_loss )
        
        lr = get_lr( self.optimizer )
        # self.scheduler.step( epoch_loss )
        self.scheduler.step()

        return epoch_loss / batch_no , epoch_accuracy_p, lr, test_losses

    def test_epoch(self, epoch_no):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        epoch_accuracy_p = 0

        with torch.no_grad():
            p_bar = tqdm(
                LeanNetRunner( self.network , self.test_data , self.trainer_params['batch_size'] ),
                desc=f'Tst {epoch_no}'
            )
            for batch_no, (batch_data, batch_out) in enumerate( p_bar ):
                batch_no += 1

                batch_loss = self.network.get_loss( batch_out.permute(0,2,1) , batch_data[:,1:] )
                epoch_loss += batch_loss.detach().item()

                if batch_no % self.trainer_params['compute_acc_every'] == 0:
                    batch_probs = self.network.get_probs( batch_out.detach() ).gather( 2 , batch_data[:,1:].unsqueeze(2) ).squeeze(2)
                    batch_accuracy = batch_probs.prod( dim=1 ).mean()
                    epoch_accuracy += batch_accuracy
                    epoch_accuracy_p = epoch_accuracy*100/(batch_no / self.trainer_params['compute_acc_every'] )

                    p_bar.set_postfix(
                        refresh=False,
                        ep_loss=f'{epoch_loss/batch_no:.4f}',
                        btc_acc = f'{batch_accuracy*100:.2f}%',
                        ep_acc=f'{epoch_accuracy_p:.2f}%'
                    )

        return epoch_loss / batch_no , epoch_accuracy_p