# Code for CVPR'21 paper:
# [Title]  - "CoLA: Weakly-Supervised Temporal Action Localization with Snippet Contrastive Learning"
# [Author] - Can Zhang*, Meng Cao, Dongming Yang, Jie Chen and Yuexian Zou
# [Github] - https://github.com/zhang-can/CoLA

import numpy as np
import os
from easydict import EasyDict as edict

cfg = edict()

# NCE Specific Parameters
cfg.PROJ_DIM = 128
cfg.NCE_WEIGHT = 0.0 # 0.5
cfg.QUEUE_SIZE = 50000
cfg.SAMPLING_RATE = 0.2
cfg.LOG_PATH = '/home/ulas/Documents/PhD/2.Codes/CoLA/experiments'
# Video Distance Specific Paramaters
cfg.PSEUDO_WEIGHT = 0.5

# Latent Representation parameters
cfg.LATENT_LOSS_WEIGHT = 1 # Really force latent representation to be similar
cfg.PRETRAIN_ENCODER_DECODER = True
cfg.PRETRAIN_BATCH_SIZE = 50
cfg.PRETRAIN_NUM_ITERS = 200
cfg.LATENT_LOSS_PRE = 1000
# CoLA Configurations
cfg.GPU_ID = '0'
cfg.LR = '[0.0001]*6000'
cfg.NUM_ITERS = len(eval(cfg.LR))
cfg.NUM_CLASSES = 20
cfg.MODAL = 'all'
cfg.FEATS_DIM = 2048 # This is the feature size x2, loader combines rbg-flow into a a single vector!
cfg.BATCH_SIZE = 16
cfg.DATA_PATH = '/home/ulas/Documents/Datasets/CoLA/data/THUMOS14'
cfg.NUM_WORKERS = 8
cfg.LAMBDA = 0.01
cfg.R_EASY = 5
cfg.R_HARD = 20
cfg.m = 3
cfg.M = 6
cfg.TEST_FREQ = 50
cfg.PRINT_FREQ = 20
cfg.CLASS_THRESH = 0.2
cfg.NMS_THRESH = 0.6
cfg.CAS_THRESH = np.arange(0.0, 0.25, 0.025)
cfg.ANESS_THRESH = np.arange(0.1, 0.925, 0.025)
cfg.TIOU_THRESH = np.linspace(0.1, 0.7, 7)
cfg.UP_SCALE = 24
cfg.GT_PATH = os.path.join(cfg.DATA_PATH, 'gt.json')
cfg.SEED = 0
cfg.FEATS_FPS = 25
cfg.NUM_SEGMENTS = 750
cfg.CLASS_DICT = {'BaseballPitch': 0, 'BasketballDunk': 1, 'Billiards': 2, 
                  'CleanAndJerk': 3, 'CliffDiving': 4, 'CricketBowling': 5, 
                  'CricketShot': 6, 'Diving': 7, 'FrisbeeCatch': 8, 
                  'GolfSwing': 9, 'HammerThrow': 10, 'HighJump': 11, 
                  'JavelinThrow': 12, 'LongJump': 13, 'PoleVault': 14, 
                  'Shotput': 15, 'SoccerPenalty': 16, 'TennisSwing': 17, 
                  'ThrowDiscus': 18, 'VolleyballSpiking': 19}
