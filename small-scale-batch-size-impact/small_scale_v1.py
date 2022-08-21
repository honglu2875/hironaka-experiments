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
from hironaka.trainer.player_modules import ChooseFirstAgentModule




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

    
def experiment(device_num, config):
    device = torch.device(f'cuda:{device_num}') if config['use_cuda'] else torch.device('cpu')
    exp_cfg = config.copy()
    
    #exp_cfg['host']['batch_size'] = 128
    exp_cfg['host']['optim']['lr'] = 0.00000001

    for i in range(5):
        for j in range(3):
            exp_cfg['host']['batch_size'] = 256 * (4**j)
            exp_cfg['host']['net_arch'] = [{'repeat': 2*(i+1), 'net_arch': ['r256']}]  # Override with small network

            print(f"starting batch size={exp_cfg['host']['batch_size']}, layer size={2*(i+1)}")
            # Experiment 1: host against choose first
            trainer = DQNTrainer(exp_cfg, device_num=device_num, agent_net=ChooseFirstAgentModule(3, 20, device))
            with open(f'exp3_{i}_{j}.yaml','w') as f:
                yaml.dump(exp_cfg, f)

            logger = trainer.logger
            logger.setLevel(logging.INFO)
            if not logger.hasHandlers():
                logger.addHandler(logging.StreamHandler(sys.stdout))
                logger.addHandler(logging.FileHandler('out1.log'))
            for k in range(20):
                trainer.train(5000, evaluation_interval=999, players=('host',))
                trainer.save(f"models/{trainer.version_string}_EXP3_{device_num}_{i}_{j}_ckpt_{k}.pt")

            print('Final')
            print(trainer.evaluate_rho())
    
def main(config_path: str = 'small_scale_v1.yml'): 
    config = DQNTrainer.load_yaml(config_path)
    batch_size = config['host']['batch_size']
    to_be_run = [experiment]
    processes = []
    for exp in to_be_run:
        processes.append(Process(target=exp, args=(0, config)))
        processes[-1].start()
    for i in range(len(processes)):
        processes[i].join()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
