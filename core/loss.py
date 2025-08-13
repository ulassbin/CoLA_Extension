# Code for CVPR'21 paper:
# [Title]  - "CoLA: Weakly-Supervised Temporal Action Localization with Snippet Contrastive Learning"
# [Author] - Can Zhang*, Meng Cao, Dongming Yang, Jie Chen and Yuexian Zou
# [Github] - https://github.com/zhang-can/CoLA

import torch
import torch.nn as nn

from NCELoss.NNIICLUV_Tests.loss import InfoNCELoss, KLDivLoss, LatentLossMasked, KLDivergencePseudoLoss

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
        print('Logits shape ', logits.shape)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = self.ce_criterion(logits, labels)

        return loss

    def forward(self, contrast_pairs):
        for keys, vals in contrast_pairs.items():
            print('Keys are ', keys)
        print(f"HA {contrast_pairs['HA'].shape}, EA {contrast_pairs['EA'].shape}, EB {contrast_pairs['EB'].shape}")
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
        

class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.mse_criterion = nn.MSELoss()
    
    def forward(self, base_feature, decoded_feature):
        loss = self.mse_criterion(base_feature, decoded_feature)
        return loss


class TotalLoss(nn.Module):
    def __init__(self, cfg):
        super(TotalLoss, self).__init__()
        self.kl_latent = cfg.KL_PSEUDO
        self.action_criterion = ActionLoss()
        self.snico_criterion = SniCoLoss()
        self.nce_criterion = InfoNCELoss()
        if self.kl_latent:
            self.vid_pseudo_loss = KLDivergencePseudoLoss() #LatentLossMasked() # Masked() #VidPseudoLoss()
        else:
            self.vid_pseudo_loss = LatentLossMasked()
        self.latent_loss = LatentLoss()
        self.kldiv_loss = KLDivLoss()
        self.nce_weight = cfg.NCE_WEIGHT
        self.pseudo_weight = cfg.PSEUDO_WEIGHT
        self.latent_weight = cfg.LATENT_LOSS_WEIGHT
        self.kldiv_weight = cfg.KLDIV_LOSS
        self.kldiv_inter_scaling = cfg.KLDIV_INTER_SCALING
        self.action_weight = cfg.ACTION_LOSS
        self.snico_weight = cfg.SNICO_LOSS # prev 0.01 hardcoded


    def forward(self, video_scores, label, contrast_pairs, sampled_embeddings, positives, negatives, pseudo_video_scores, enc_decoder_embeddings, intra_params, inter_params):
        input_feature, decoded_inter, decoded_intra = enc_decoder_embeddings
        #print("video_scores", video_scores.shape, video_scores.min(), video_scores.max())
        #print("labels", label.shape, label.min(), label.max())
        loss_cls = self.action_criterion(video_scores, label) # Classification Loss
        loss_snico = self.snico_criterion(contrast_pairs) # CoLa Contrastive Loss
        loss_nce = self.nce_criterion(sampled_embeddings, positives, negatives)
        #print(f'Vid scores {video_scores.shape}, pseudo {pseudo_video_scores.shape}')
        loss_pseudo = self.vid_pseudo_loss(video_scores, pseudo_video_scores) # Video Pseudo Label Loss
        loss_latent_inter = self.latent_loss(input_feature, decoded_inter)
        loss_latent_intra = self.latent_loss(input_feature, decoded_intra)
        batch, time, feats = intra_params[0].shape
        # Reshape intra_params and inter_params to match the expected dimensions
        kldiv_intra = self.kldiv_loss(intra_params[0].reshape(-1, feats), intra_params[1].reshape(-1,feats)) # param0 is mu, param1 is logvar # (B*Txfeats) # framewise representation
        kldiv_inter = self.kldiv_loss(inter_params[0].reshape(batch, -1), inter_params[1].reshape(batch,-1)) / time # param 0 is mu, param1 is logvar # (BxT*feats) # video wise representation
        
        loss_total = self.action_weight * loss_cls + self.snico_weight * loss_snico + self.nce_weight * loss_nce + self.pseudo_weight * loss_pseudo

        loss_total += self.latent_weight * (loss_latent_inter + loss_latent_intra)
        loss_total += self.kldiv_weight * (kldiv_intra + self.kldiv_inter_scaling * kldiv_inter) / 2.0

        loss_dict = {
            'Loss/Total': loss_total,
            'Loss/Action': loss_cls,
            'Loss/SniCo': loss_snico,
            'Loss/NCE': loss_nce,
            'Loss/Pseudo': loss_pseudo,
            'Loss/LatentInter': loss_latent_inter,
            'Loss/LatentIntra': loss_latent_intra,
            'Loss/KLDivIntra': kldiv_intra,
            'Loss/KLDivInter': kldiv_inter
        }
        #for keys, vals in loss_dict.items():
        #    print(f'{keys}: {vals}')
        return loss_total, loss_dict
