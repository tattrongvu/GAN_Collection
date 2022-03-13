import os
import json
import datetime
import random
import torch
import argparse

import numpy as np
from collections import namedtuple

from CycleGAN import CycleGAN

def get_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--config', type=str, default='config.json', help='the configuration file name')
    parser.add_argument('--cuda', type=int, default=0, help='which gpu should be use')
    args = parser.parse_args()

    return args

def train(config_name:str = "config.json",cuda:int=0):
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), config_name), mode="r") as f:
        args = json.load(
            f, object_hook=lambda d: namedtuple("X", d.keys())(*d.values())
        )
    
    device = torch.device("cuda:"+str(cuda) if torch.cuda.is_available() else "cpu")

    print("Training on: ",device)
    print("Model name: ", args.model_name)
    print("Dataset name: ", args.dataset_name)

    agent = CycleGAN(args = args, device=device)
    agent.learn()

if __name__ == "__main__":
    args = get_args()
    train(cuda=args.cuda,config_name=args.config)