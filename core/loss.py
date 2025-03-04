# Code for CVPR'21 paper:
# [Title]  - "CoLA: Weakly-Supervised Temporal Action Localization with Snippet Contrastive Learning"
# [Author] - Can Zhang*, Meng Cao, Dongming Yang, Jie Chen and Yuexian Zou
# [Github] - https://github.com/zhang-can/CoLA

import torch
import torch.nn as nn

from NCELoss.NNIICLUV_Tests.loss import InfoNCELoss

class ActionLoss(nn.Module):
    def __init__(self):
        super(ActionLoss, self).__init__()
        self.bce_criterion = nn.BCELoss()

    def forward(self, video_scores, label):
        label = label / torch.sum(label, dim=1, keepdim=True)
        loss = self.bce_criterion(video_scores, label)
        return loss

class SniCoLoss(nn.Module):
    def __init__(self):
        super(SniCoLoss, self).__init__()
        self.ce_criterion = nn.CrossEntropyLoss()

    def NCE(self, q, k, neg, T=0.07):
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        neg = neg.permute(0,2,1)
        neg = nn.functional.normalize(neg, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,nck->nk', [q, neg])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = self.ce_criterion(logits, labels)

        return loss

    def forward(self, contrast_pairs):

        HA_refinement = self.NCE(
            torch.mean(contrast_pairs['HA'], 1), 
            torch.mean(contrast_pairs['EA'], 1), 
            contrast_pairs['EB']
        )

        HB_refinement = self.NCE(
            torch.mean(contrast_pairs['HB'], 1), 
            torch.mean(contrast_pairs['EB'], 1), 
            contrast_pairs['EA']
        )

        loss = HA_refinement + HB_refinement
        return loss
        

class VidPseudoLoss(nn.Module):
    def __init__(self):
        super(VidPseudoLoss, self).__init__()
        self.bce_criterion = nn.BCELoss()
    
    def forward(self, video_scores, pseudo_label):
        loss = self.bce_criterion(video_scores, pseudo_label)
        return loss
        

class LatentLoss(nn.Module):
    def __init__(self):
        super(LatentLoss, self).__init__()
        self.mse_criterion = nn.MSELoss()
    
    def forward(self, base_feature, decoded_feature):
        loss = self.mse_criterion(base_feature, decoded_feature)
        return loss


class TotalLoss(nn.Module):
    def __init__(self, cfg):
        super(TotalLoss, self).__init__()
        self.action_criterion = ActionLoss()
        self.snico_criterion = SniCoLoss()
        self.nce_criterion = InfoNCELoss()
        self.vid_pseudo_loss = VidPseudoLoss()
        self.latent_loss = LatentLoss()
        self.nce_weight = cfg.NCE_WEIGHT
        self.pseudo_weight = cfg.PSEUDO_WEIGHT
        self.latent_weight = cfg.LATENT_LOSS_WEIGHT


    def forward(self, video_scores, label, contrast_pairs, sampled_embeddings, positives, negatives, pseudo_label, enc_decoder_embeddings):
        input_feature, decoded_inter, decoded_intra = enc_decoder_embeddings
        loss_cls = self.action_criterion(video_scores, label)
        loss_snico = self.snico_criterion(contrast_pairs)
        loss_nce = self.nce_criterion(sampled_embeddings, positives, negatives)
        loss_pseudo = self.vid_pseudo_loss(video_scores, pseudo_label)
        loss_latent_inter = self.latent_loss(input_feature, decoded_inter)
        loss_latent_intra = self.latent_loss(input_feature, decoded_intra)
        loss_total = loss_cls + 0.01 * loss_snico + self.nce_weight * loss_nce + self.pseudo_weight * loss_pseudo
        
        loss_total += self.latent_weight *(loss_latent_inter + loss_latent_intra)

        loss_dict = {
            'Loss/Total': loss_total,
            'Loss/Action': loss_cls,
            'Loss/SniCo': loss_snico,
            'Loss/Intra': loss_nce,
            'Loss/Pseudo': loss_pseudo,
            'Loss/LatentInter': loss_latent_inter,
            'Loss/LatentIntra': loss_latent_intra,
            'Loss/LatentCombined': loss_latent_inter + loss_latent_intra
        }

        return loss_total, loss_dict
