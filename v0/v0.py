import argparse
import logging
import os
import pathlib
import sys
import time
import torch as th
from torch import nn
from torch.nn import functional as F
from multiprocessing import Process
from torch.utils.tensorboard import SummaryWriter

from hfai.utils import which_numa

sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()) + "/hironaka")

from hironaka.policy import NNPolicy
from hironaka.validator import HironakaValidator

import yaml
import torch
import numpy as np
import logging

from hironaka.agent import RandomAgent, ChooseFirstAgent, PolicyAgent
from hironaka.host import Zeillinger, RandomHost, PolicyHost
from hironaka.core import Points, TensorPoints
from hironaka.src import coord_list_to_binary, encode_action, decode_action, get_shape, mask_encoded_action, generate_batch_points
from hironaka.validator import HironakaValidator
from hironaka.util import FusedGame
from hironaka.trainer.DQNTrainer import DQNTrainer




logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    logger.addHandler(logging.StreamHandler(sys.stdout))

model_path = 'models'
config_file = 'config.yml'
if not os.path.exists(model_path):
    logger.info("Created 'models/'.")
    os.makedirs(model_path)
else:
    logger.warning("Model folder 'models/' already exists.")

    
def process(device_num, config_path):  
    if config_path is None:
        config_path='trainer_cfg_v0.yml'
    trainer = DQNTrainer(config_path, device_num=device_num)
    logger = trainer.logger
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        logger.addHandler(logging.StreamHandler(sys.stdout))
        
    trainer.train(100000)

    print(trainer.evaluate_rho())
    
    trainer.save(f"models/{trainer.version_string}_{device_num}.pt")

def main(config_path: str = 'trainer_cfg_v0.yml'): 
    num_device = 8
    processes = []
    for i in range(num_device):
        processes.append(Process(target=process, args=(i, config_path)))
        processes[-1].start()
    for i in range(num_device):
        processes[i].join()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
