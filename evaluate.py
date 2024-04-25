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


def main():
    # Commandline arg
    parser = argparse.ArgumentParser(description="VQA eval parser")
    parser.add_argument("--save_filepath", "-p", default="", type=str, help="file path to the saved model artifact")
    args = vars(parser.parse_args())

    # Config
    device = utils.set_device(mps=False, cuda=False)
    if device == "cuda":
        cudnn.benchmark = True

    # data
    testloader = data.get_loader(test=True)

    # model
    pretrained_model = BertModel.from_pretrained("bert-base-uncased")
    net = model.SimpleNet(pretrained_model).to(device)

    # evaluate on test dataset
    #metrics = evaluate(net, testloader)


if __name__ == '__main__':
    main()
