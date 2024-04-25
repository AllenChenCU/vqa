import sys
import os
import json
from datetime import datetime
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

import config
import data
import model
from utils import utils


class Trainer:
    def __init__(self, net, optimizer, criterion, device, tracker):
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.tracker = tracker
        self.config_as_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.iterations = 0
        self.class_weights = {
            "correct": 1, 
            "incorrect": 1/2, 
        }

    def update_learning_rate(self):
        lr = config.INITIAL_LR * 0.5**(float(self.iterations) / config.LR_HALFLIFE)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def train(self, trainloader, valloader, save_dir="logs", save_filename_prefix="test"):
        
        for epoch in tqdm(range(config.EPOCHS)):
            save_filename = f"{save_filename_prefix}_{epoch}.pth"
            #save_filename = f"{save_filename_prefix}.pth"
            save_filepath = os.path.join(save_dir, save_filename)
            _ = self._run_epoch(trainloader, epoch=epoch, train=True)
            preds, overall_accs, pos_accs, idxs, q_ids = self._run_epoch(valloader, epoch=epoch, train=False)

            results = {
                'filename': save_filepath, 
                'tracker': self.tracker.to_dict(), 
                'config': self.config_as_dict,
                'weights': self.net.state_dict(), 
                'eval': {
                    'preds': preds,  # predictions in the validation dataset
                    'overall_accs': overall_accs, # accuracies of mini-batches
                    'pos_accs': pos_accs, 
                    'idx': idxs,        # indices in the validation dataset
                    'q_ids': q_ids,     # question ids in the validation dataset
                }, 
            }
            torch.save(results, save_filepath)

    def _run_epoch(self, dataloader, epoch, train=False):
        if train:
            self.net.train()
            tracker_class = self.tracker.MovingMeanMonitor
            tracker_params = {'momentum': 0.99}
            prefix = "train"
        else:
            self.net.eval()
            tracker_class = self.tracker.MeanMonitor
            tracker_params = {} 
            prefix = "val"
        
        tq = tqdm(dataloader, desc=f"{prefix} Epoch: {epoch}", ncols=0)
        loss_tracker = self.tracker.track(f"{prefix}_loss", tracker_class(**tracker_params))
        overall_acc_tracker = self.tracker.track(f"{prefix}_overall_acc", tracker_class(**tracker_params))
        pos_acc_tracker = self.tracker.track(f"{prefix}_pos_acc", tracker_class(**tracker_params))

        preds = [] # all prediction outputs
        overall_accs = []    # accuracies of mini-batches
        pos_accs = []
        idxs = []    # indices of the validation dataset
        q_ids = []  # question_ids of the validation dataset

        for v, q, c, a, idx, q_id in tq:

            # forward
            wrapped_input = self.tokenizer(
                text=q, 
                text_pair=c, 
                add_special_tokens=True, 
                truncation=False, 
                padding=True, 
                return_tensors="pt"
            )

            v = v.to(self.device)
            for k, _ in wrapped_input.items():
                wrapped_input[k] = wrapped_input[k].to(self.device)
            output = self.net(v, wrapped_input).view(-1)
            a = a.type(torch.FloatTensor).to(self.device)
            self.criterion.weight = a * self.class_weights["correct"] + (1-a)*self.class_weights["incorrect"]
            loss = self.criterion(output, a)

            # calc acc
            overall_acc = (output.round() == a).float().mean()

            agree = (a == output.round()).type(torch.IntTensor)
            indices_agree = torch.nonzero(a).view(-1) # convert mask to indices
            pos_agree = agree[indices_agree]     # accuracy for positive examples only
            pos_acc = pos_agree.float().mean()


            if train:
                # update learning rate
                self.update_learning_rate()
                self.iterations += 1

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            loss_tracker.append(loss.detach().item())
            overall_acc_tracker.append(overall_acc)
            pos_acc_tracker.append(pos_acc)
            tq.set_postfix(
                loss="{:.4f}".format(loss), 
                overall_acc="{:.4f}".format(overall_acc),
                pos_acc="{:.4f}".format(pos_acc),
            )
            preds.append(output.view(-1))
            overall_accs.append(overall_acc.view(-1))
            pos_accs.append(pos_acc.view(-1))
            idxs.append(idx.view(-1).clone())
            q_ids.append(q_id.view(-1).clone())
        
        preds = torch.cat(preds, dim=0)
        overall_accs = torch.cat(overall_accs, dim=0)
        pos_accs = torch.cat(pos_accs, dim=0)
        idxs = torch.cat(idxs, dim=0)
        q_ids = torch.cat(q_ids, dim=0)
        return preds, overall_accs, pos_accs, idxs, q_ids


def main():
    parser = argparse.ArgumentParser(description="VQA trainer parser")
    parser.add_argument("--name", "-n", default="test", type=str, help="Name of the training run")
    args = vars(parser.parse_args())

    # Config
    time_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    target_filename_prefix = f"{args['name']}_{time_now}"
    target_dir = "logs"
    device = utils.set_device(mps=False, cuda=config.CUDA)
    if device == "cuda":
        cudnn.benchmark = True

    # data
    trainloader = data.get_loader(train=True)
    valloader = data.get_loader(val=True)

    # model
    pretrained_model = BertModel.from_pretrained("bert-base-uncased")
    net = model.SimpleNet(pretrained_model).to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    tracker = utils.Tracker()
    trainer = Trainer(net, optimizer, criterion, device, tracker)
    trainer.train(trainloader, valloader, target_dir, target_filename_prefix)


if __name__ == '__main__':
    main()
