# Code for CVPR'21 paper:
# [Title]  - "CoLA: Weakly-Supervised Temporal Action Localization with Snippet Contrastive Learning"
# [Author] - Can Zhang*, Meng Cao, Dongming Yang, Jie Chen and Yuexian Zou
# [Github] - https://github.com/zhang-can/CoLA

import os
import sys
import time
import copy
import json
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import core.utils as utils
from core.model import CoLA
from core.loss import TotalLoss, LatentLoss 
from core.config import cfg
from core.utils import AverageMeter
from core.dataset import NpyFeature
from torch.utils.tensorboard import SummaryWriter
from eval.eval_detection import ANETdetection
from terminaltables import AsciiTable

from NCELoss.NNIICLUV_Tests.custom_queue import Queue


class Trainer:
  def __init__(self, cfg):
      self.cfg = cfg
      self.queue = Queue(queue_size=cfg.QUEUE_SIZE, embedding_dim=cfg.PROJ_DIM, device='cuda')
      self.nn_queue = Queue(queue_size=cfg.QUEUE_SIZE, embedding_dim=cfg.PROJ_DIM, device='cuda')
      self.initialized = False

  def initialize(self, embeddings):
      print('Initializeing with ', embeddings.shape)
      self.queue.enqueue(embeddings)
      self.initialized = True
      return

  def get_positives_video_distance(self, full_embeddings, temporal, embedding_dim, debug=False):
      # In this function we will get the positives by using fft based distance calculation
      batch_size, temporal, embedding_dim = full_embeddings.shape
      polled_vids = batch_size
      vid_embeddings, vid_indices = self.queue.find_nearest_vids(full_embeddings)# Implement this
      #if vid_labels is not None:
      #    vid_labels = vid_labels.reshape(batch_size, 3)
      return vid_embeddings, vid_indices#, vid_labels
  
  
  def get_positives(self, intra_embeddings, temporal, embedding_dim, debug=False): # In future get K input
    batch_size, temporal, embedding_dim = intra_embeddings.shape
    intra_embeddings = intra_embeddings.reshape(batch_size * temporal, embedding_dim)
    nn_indices, nn_embeddings, nn_labels = self.queue.find_nearest_neighbors(intra_embeddings)
    nn_embeddings = nn_embeddings.reshape(batch_size, temporal, embedding_dim)
    nn_indices = nn_indices.reshape(batch_size, temporal)
    if nn_labels is not None:
      nn_labels = nn_labels.reshape(batch_size, temporal, 3)
    return nn_indices, nn_embeddings, nn_labels
  
  def sample_embeddings(self, embeddings):
      batch_size, t, feature_dim = embeddings.shape  # Assuming fixed t
      new_size = int(self.cfg.SAMPLING_RATE * t)
      sample_indexes = torch.randint(0, t, (batch_size, new_size), device=embeddings.device)
      # Use advanced indexing to preserve gradients
      sampled = embeddings[torch.arange(batch_size).unsqueeze(1), sample_indexes]
      return sampled  # Keeps gradients


  def pretrain_encoder_decoder_step(self, net, loader_iter, optimizer, criterion, writer, step):
      net.train()
      data, label, _, _, _ = next(loader_iter)
      data = data.cuda()
      label = label.cuda()
      optimizer.zero_grad()
      video_scores, contrast_pairs, _, _, all_embeddings = net(data)
      criterion = LatentLoss()
      decoded_inter = all_embeddings[2]
      decoded_intra = all_embeddings[3]
      cost = cfg.LATENT_LOSS_PRE * (criterion(data, decoded_inter) + criterion(data, decoded_intra))/2.0
      cost.backward()
      optimizer.step()
      writer.add_scalar('PRE_Latent Loss', cost.cpu().item(), step)
      return cost

  def train_one_step(self, net, loader_iter, optimizer, criterion, writter, step):
      net.train()
      
      data, label, _, _, _ = next(loader_iter)
      data = data.cuda()
      label = label.cuda()

      optimizer.zero_grad()
      video_scores, contrast_pairs, _, _, all_embeddings = net(data)
      # all_embbeddings are [intra_embeddings, inter_embeddings, decoded_inter, decoded_intra]
      # Sample intra_embeddings
      intra_embeddings = all_embeddings[0]
      embedding_targets = self.sample_embeddings(intra_embeddings)
      #print('Embedding Targets {}, Intra Embeddings {}'.format(embedding_targets.shape, intra_embeddings.shape))

      if not self.initialized:
          self.initialize(embedding_targets)
      # Snippet Contrastive Learning
      positive_indices, positives, positive_labels = self.get_positives(embedding_targets, cfg.NUM_SEGMENTS, cfg.PROJ_DIM)
      negatives, negative_indexes = self.queue.getNegatives(positive_indices)
      # Video Contrastive Learning
      vid_positives, vid_positives_indices = self.get_positives_video_distance(intra_embeddings, cfg.NUM_SEGMENTS, cfg.PROJ_DIM)
      with torch.no_grad():
          pseudo_labels, _, _  = net.forward_with_embeddings(vid_positives) # The issue here is vid_positives are 128 feature size not 2048
                                                            # So we cant use it directly to feed it to the model. But that was a good try.
      #print('Embeddins {}, Positives {}, Negatives {}'.format(intra_embeddings.shape, positives.shape, negatives.shape))
      cost, loss = criterion(video_scores, label, contrast_pairs, embedding_targets, positives, negatives, pseudo_labels,
                             [data, all_embeddings[2], all_embeddings[3]])
      
      cost.backward()
      optimizer.step()
      self.queue.enqueue(intra_embeddings)
      for key in loss.keys():
          writter.add_scalar(key, loss[key].cpu().item(), step)
      return cost

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU_ID
    worker_init_fn = None
    if cfg.SEED >= 0:
        utils.set_seed(cfg.SEED)
        worker_init_fn = np.random.seed(cfg.SEED)

    utils.set_path(cfg)
    utils.save_config(cfg)

    net = CoLA(cfg)
    net = net.cuda()

    print('Log Path: {}'.format(cfg.LOG_PATH))
    train_loader = torch.utils.data.DataLoader(
        NpyFeature(data_path=cfg.DATA_PATH, mode='train',
                        modal=cfg.MODAL, feature_fps=cfg.FEATS_FPS,
                        num_segments=cfg.NUM_SEGMENTS, supervision='weak',
                        class_dict=cfg.CLASS_DICT, seed=cfg.SEED, sampling='random'),
            batch_size=cfg.BATCH_SIZE,
            shuffle=True, num_workers=cfg.NUM_WORKERS,
            worker_init_fn=worker_init_fn)

    test_loader = torch.utils.data.DataLoader(
        NpyFeature(data_path=cfg.DATA_PATH, mode='test',
                        modal=cfg.MODAL, feature_fps=cfg.FEATS_FPS,
                        num_segments=cfg.NUM_SEGMENTS, supervision='weak',
                        class_dict=cfg.CLASS_DICT, seed=cfg.SEED, sampling='uniform'),
            batch_size=1,
            shuffle=False, num_workers=cfg.NUM_WORKERS,
            worker_init_fn=worker_init_fn)

    test_info = {"step": [], "test_acc": [], "average_mAP": [],
                "mAP@0.1": [], "mAP@0.2": [], "mAP@0.3": [], 
                "mAP@0.4": [], "mAP@0.5": [], "mAP@0.6": [],
                "mAP@0.7": []}
    
    best_mAP = -1

    criterion = TotalLoss(cfg)
  
    cfg.LR = eval(cfg.LR)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.LR[0],
        betas=(0.9, 0.999), weight_decay=0.0005)

    if cfg.MODE == 'test':
        _, _ = test_all(net, cfg, test_loader, test_info, 0, None, cfg.MODEL_FILE)
        utils.save_best_record_thumos(test_info, 
            os.path.join(cfg.OUTPUT_PATH, "best_results.txt"))
        print(utils.table_format(test_info, cfg.TIOU_THRESH, '[CoLA] THUMOS\'14 Performance'))
        return
    else:
        writter = SummaryWriter(cfg.LOG_PATH)
        
    print('=> test frequency: {} steps'.format(cfg.TEST_FREQ))
    print('=> start training...')
    trainer = Trainer(cfg)


    pre_train_loader = torch.utils.data.DataLoader(
        NpyFeature(data_path=cfg.DATA_PATH, mode='train',
                        modal=cfg.MODAL, feature_fps=cfg.FEATS_FPS,
                        num_segments=cfg.NUM_SEGMENTS, supervision='weak',
                        class_dict=cfg.CLASS_DICT, seed=cfg.SEED, sampling='random'),
            batch_size=cfg.PRETRAIN_BATCH_SIZE,
            shuffle=True, num_workers=cfg.NUM_WORKERS,
            worker_init_fn=worker_init_fn)



    if cfg.PRETRAIN_ENCODER_DECODER:
        for step in range(1, cfg.PRETRAIN_NUM_ITERS + 1):
            if step > 1 and cfg.LR[step - 1] != cfg.LR[step - 2]:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = cfg.LR[step - 1]

            if (step - 1) % len(pre_train_loader) == 0:
                loader_iter = iter(pre_train_loader)

            batch_time = AverageMeter()
            losses = AverageMeter()
            end = time.time()
            cost = trainer.pretrain_encoder_decoder_step(net, loader_iter, optimizer, criterion, writter, step)
            losses.update(cost.item(), cfg.BATCH_SIZE)
            batch_time.update(time.time() - end)
            end = time.time()
            if step == 1 or step % cfg.PRINT_FREQ == 0:
                print(('PRETRAIN: Step: [{0:04d}/{1}]\t' \
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    step, cfg.PRETRAIN_NUM_ITERS, batch_time=batch_time, loss=losses)))
    # How to free memory here?
    del pre_train_loader
    del loader_iter
    torch.cuda.empty_cache()
    print('=> pretraining done...')
    print('=> start training...')

    for step in range(1, cfg.NUM_ITERS + 1):
        if step > 1 and cfg.LR[step - 1] != cfg.LR[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = cfg.LR[step - 1]

        if (step - 1) % len(train_loader) == 0:
            loader_iter = iter(train_loader)

        batch_time = AverageMeter()
        losses = AverageMeter()
        
        end = time.time()
        cost = trainer.train_one_step(net, loader_iter, optimizer, criterion, writter, step,)
        losses.update(cost.item(), cfg.BATCH_SIZE)
        batch_time.update(time.time() - end)
        end = time.time()
        if step == 1 or step % cfg.PRINT_FREQ == 0:
            print(('Step: [{0:04d}/{1}]\t' \
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    step, cfg.NUM_ITERS, batch_time=batch_time, loss=losses)))
            
        if step > -1 and step % cfg.TEST_FREQ == 0:

            mAP_50, mAP_AVG = test_all(net, cfg, test_loader, test_info, step, writter)

            if test_info["average_mAP"][-1] > best_mAP:
                best_mAP = test_info["average_mAP"][-1]
                best_test_info = copy.deepcopy(test_info)

                utils.save_best_record_thumos(test_info, 
                    os.path.join(cfg.OUTPUT_PATH, "best_results.txt"))

                torch.save(net.state_dict(), os.path.join(cfg.MODEL_PATH, \
                    "model_best.pth.tar"))

            print(('- Test result: \t' \
                   'mAP@0.5 {mAP_50:.2%}\t' \
                   'mAP@AVG {mAP_AVG:.2%} (best: {best_mAP:.2%})'.format(
                   mAP_50=mAP_50, mAP_AVG=mAP_AVG, best_mAP=best_mAP)))

    print(utils.table_format(best_test_info, cfg.TIOU_THRESH, '[CoLA] THUMOS\'14 Performance'))

@torch.no_grad()
def test_all(net, cfg, test_loader, test_info, step, writter=None, model_file=None):
    net.eval()

    if model_file:
        print('=> loading model: {}'.format(model_file))
        net.load_state_dict(torch.load(model_file))
        print('=> tesing model...')

    final_res = {'method': '[CoLA] https://github.com/zhang-can/CoLA', 'results': {}}
    
    acc = AverageMeter()

    for data, label, _, vid, vid_num_seg in test_loader:
        data, label = data.cuda(), label.cuda()
        vid_num_seg = vid_num_seg[0].cpu().item()

        video_scores, _, actionness, cas, all_embeddings = net(data) # No use for embeddings here.

        label_np = label.cpu().data.numpy()
        score_np = video_scores[0].cpu().data.numpy()
        
        pred_np = np.where(score_np < cfg.CLASS_THRESH, 0, 1)
        correct_pred = np.sum(label_np == pred_np, axis=1)
        acc.update(float(np.sum((correct_pred == cfg.NUM_CLASSES))), correct_pred.shape[0])

        pred = np.where(score_np >= cfg.CLASS_THRESH)[0]
        if len(pred) == 0:
            pred = np.array([np.argmax(score_np)])
        
        cas_pred = utils.get_pred_activations(cas, pred, cfg)
        aness_pred = utils.get_pred_activations(actionness, pred, cfg)
        proposal_dict = utils.get_proposal_dict(cas_pred, aness_pred, pred, score_np, vid_num_seg, cfg)

        final_proposals = [utils.nms(v, cfg.NMS_THRESH) for _,v in proposal_dict.items()]
        final_res['results'][vid[0]] = utils.result2json(final_proposals, cfg.CLASS_DICT)

    json_path = os.path.join(cfg.OUTPUT_PATH, 'result.json')
    json.dump(final_res, open(json_path, 'w'))
    
    anet_detection = ANETdetection(cfg.GT_PATH, json_path,
                                subset='test', tiou_thresholds=cfg.TIOU_THRESH,
                                verbose=False, check_status=False)
    mAP, average_mAP = anet_detection.evaluate()

    if writter:
        writter.add_scalar('Test Performance/Accuracy', acc.avg, step)
        writter.add_scalar('Test Performance/mAP@AVG', average_mAP, step)
        for i in range(cfg.TIOU_THRESH.shape[0]):
            writter.add_scalar('mAP@tIOU/mAP@{:.1f}'.format(cfg.TIOU_THRESH[i]), mAP[i], step)

    test_info["step"].append(step)
    test_info["test_acc"].append(acc.avg)
    test_info["average_mAP"].append(average_mAP)

    for i in range(cfg.TIOU_THRESH.shape[0]):
        test_info["mAP@{:.1f}".format(cfg.TIOU_THRESH[i])].append(mAP[i])
    return test_info['mAP@0.5'][-1], average_mAP

if __name__ == "__main__":
    assert len(sys.argv)>=2 and sys.argv[1] in ['train', 'test'], 'Please set mode (choices: [train] or [test])'
    cfg.MODE = sys.argv[1]
    if cfg.MODE == 'test':
        assert len(sys.argv) == 3, 'Please set model path'
        cfg.MODEL_FILE = sys.argv[2]
    print(AsciiTable([['CoLA - Compare to Localize Actions']]).table)
    main()