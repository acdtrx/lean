import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
# import statistics as stats

from tqdm import tqdm

# trainer class
class Trainer():
    def __init__(self, device, network, train_data, test_data, trainer_params):
        self.network = network
        self.trainer_params = trainer_params
        self.device = device

        train_ds = TensorDataset( train_data )
        test_ds = TensorDataset( test_data )

        self.train_dl = DataLoader( train_ds , trainer_params['batch_size'] , False )
        self.test_dl = DataLoader( test_ds , trainer_params['batch_size'] , False )

        self.optimizer = optim.Adam( self.network.parameters() , lr=self.trainer_params['lr'] )
        self.scheduler = lr_scheduler.ReduceLROnPlateau( self.optimizer , mode='min' , factor=0.1 , threshold=0.1 , patience=1 , verbose=True )

    def train_epoch(self, epoch_no):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        epoch_accuracy_p = 0

        p_bar = tqdm( self.train_dl , desc=f'Trn {epoch_no}' , mininterval=1 , leave=True , dynamic_ncols=True )
        for batch_no, (train_data,) in enumerate(p_bar):
            # prepare ground truth
            train_data = train_data.to(self.device)

            hs , _ = self.network( train_data[:,:-1] )
            out = self.network.get_logits( hs )

            self.optimizer.zero_grad()
            batch_loss = self.network.get_loss( out.permute(0,2,1) , train_data[:,1:] )
            batch_loss.backward()
            self.optimizer.step()

            epoch_loss += batch_loss.detach().item()

            if batch_no % self.trainer_params['compute_acc_every'] == 0:
                batch_probs = self.network.get_probs( out.detach() / 0.8 ).gather( 2 , train_data[:,1:].unsqueeze(2) ).squeeze(2)
                batch_accuracy = batch_probs.prod( dim=1 ).mean()
                epoch_accuracy += batch_accuracy
                epoch_accuracy_p = epoch_accuracy*100/(1 + batch_no / self.trainer_params['compute_acc_every'] )

                p_bar.set_postfix(
                    refresh=False,
                    ep_loss=f'{epoch_loss/(batch_no+1):.4f}',
                    btc_acc = f'{batch_accuracy*100:.2f}%',
                    ep_acc=f'{epoch_accuracy_p:.2f}%'
                )
        
        self.scheduler.step( epoch_loss )

        return epoch_loss / ( batch_no + 1 ) , epoch_accuracy_p

    def test_epoch(self, epoch_no):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        epoch_accuracy_p = 0

        with torch.no_grad():
            p_bar = tqdm( self.test_dl , desc=f'Tst {epoch_no}' , mininterval=1 , leave=True , dynamic_ncols=True )
            for batch_no, (test_data,) in enumerate(p_bar):
                # prepare ground truth
                test_data = test_data.to(self.device)

                hs , _ = self.network( test_data[:,:-1] )
                out = self.network.get_logits( hs )

                batch_loss = self.network.get_loss( out.permute(0,2,1) , test_data[:,1:] )
                epoch_loss += batch_loss.detach().item()

                if batch_no % self.trainer_params['compute_acc_every'] == 0:
                    batch_probs = self.network.get_probs( out.detach() / 0.8 ).gather( 2 , test_data[:,1:].unsqueeze(2) ).squeeze(2)
                    batch_accuracy = batch_probs.prod( dim=1 ).mean()
                    epoch_accuracy += batch_accuracy
                    epoch_accuracy_p = epoch_accuracy*100/(1 + batch_no / self.trainer_params['compute_acc_every'] )

                    p_bar.set_postfix(
                        refresh=False,
                        ep_loss=f'{epoch_loss/(batch_no+1):.4f}',
                        btc_acc = f'{batch_accuracy*100:.2f}%',
                        ep_acc=f'{epoch_accuracy_p:.2f}%'
                    )

        return epoch_loss / ( batch_no + 1 ) , epoch_accuracy_p