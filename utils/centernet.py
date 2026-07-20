import torch
import torch.nn.functional as F
import numpy as np

def _neg_loss(preds, cengt):
    loss = 0
    for i in range(preds.shape[0]):
        gt = cengt[i, :, :]
        pos_inds = gt.eq(1)
        neg_inds = gt.lt(1)
        pred = preds[i, :, :, :].squeeze()
        neg_weights = torch.pow(1 - gt[neg_inds], 4)
        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]
        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights
        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

def _nms(heat, kernel=1):
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _topk(scores, ins_res, K=20, CosThresh=0.5):
    batch, cat, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)
    topk_clses = torch.div(topk_inds, height * width, rounding_mode='trunc').int()
    topk_inds = topk_inds % (height * width)
    topk_ys = torch.div(topk_inds, width, rounding_mode='trunc').float()
    topk_xs = (topk_inds % width).int().float()
    topk_feature = np.zeros([K, ins_res.shape[2]])
    for m in range(topk_ys.shape[1]):
        tempfeature = ins_res[topk_ys[0, m].int(), topk_xs[0, m].int(), :]
        topk_feature[m, :] = tempfeature
    matrix = np.matmul(topk_feature, topk_feature.transpose())
    proposal_center = []
    proposal_x = []
    proposal_y = []
    stop = matrix.shape[0]
    for mm in range(0, stop):
        if matrix[mm, mm] != 0:
            a = matrix[mm, :]
            b = np.where(a >= CosThresh)
            maxidx = np.argmax(topk_scores[0, b].detach().cpu().numpy())
            propsal = b[0][maxidx]
            if propsal == mm:
                proposal_center.append(topk_feature[mm, :])
                proposal_x.append(topk_xs[0, mm])
                proposal_y.append(topk_ys[0, mm])
    proposal_x = torch.stack(proposal_x).detach().cpu().numpy().astype(int)
    proposal_y = torch.stack(proposal_y).detach().cpu().numpy().astype(int)
    return topk_scores, topk_inds, topk_clses, proposal_y, proposal_x, proposal_center
