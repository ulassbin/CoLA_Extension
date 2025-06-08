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
cfg.NCE_WEIGHT = 0.0 # 0.1 # To scale to Action loss this should be around 2.5 in mag. so 0.25 is 10% prev good performance was 0.1
cfg.QUEUE_SIZE = 200000
cfg.SAMPLING_RATE = 0.2
cfg.SAMPLED_VID_NUM = 50
cfg.sampled_vid_num = cfg.SAMPLED_VID_NUM
cfg.LOG_PATH = '/abyss/home/forked_CoLa/CoLA_Extension/experiments/'
# Video Distance Specific Paramaters
cfg.KL_PSEUDO = True
cfg.PSEUDO_WEIGHT = 0.0 #1.0 #10 of action loss atm. # Keep it low...
cfg.FFT_K = 10 # Number of nearest neighbors to consider in the frequency domain
# Latent train in between
cfg.LATENT_TRAIN_BETWEEN = False
cfg.KLDIV_INTER_SCALING = 0.0001 # because inter/intra = 1k approx
cfg.KLDIV_LOSS = 1.0 # KL Divergence loss weight
#cfg.KLDIV_LOSS_SCALING=0.1
cfg.ACTION_LOSS = 1.0
# Latent Representation parameters
cfg.LATENT_LOSS_WEIGHT = 1.0 # Really force latent representation to be similar
cfg.PRETRAIN_ENCODER_DECODER = True
cfg.PRETRAIN_BATCH_SIZE = 100
cfg.PRETRAIN_NUM_ITERS = 100 # 600 # 1000
cfg.PRETRAIN_LR = 0.01 # 10 times higher
cfg.LATENT_LOSS_PRE = 1.0
# CoLA Configurations
cfg.GPU_ID = '0'
cfg.LR = '[0.001]*60000'
cfg.NUM_ITERS = len(eval(cfg.LR))
cfg.NUM_CLASSES = 20
cfg.MODAL = 'all'
cfg.FEATS_DIM = 2048 # This is the feature size x2, loader combines rbg-flow into a a single vector!
cfg.BATCH_SIZE = 50
cfg.DATA_PATH = '/abyss/home/THUMOS14'
cfg.NUM_WORKERS = 8
cfg.LAMBDA = 0.01
cfg.R_EASY = 5
cfg.R_HARD = 20
cfg.m = 3
cfg.M = 6
cfg.TEST_FREQ = 19
cfg.PRINT_FREQ = 5
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
