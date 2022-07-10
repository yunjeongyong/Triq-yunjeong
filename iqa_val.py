from iqa_args import args
import numpy as np
import scipy.stats
import torch
import torch.nn as nn


def valid(model, testloader):
    model.eval()
    results = []
    test_loss = []

    for img_id, (img, dmos) in enumerate(testloader):

        img = torch.tensor(img, device='cuda', requires_grad=False, dtype=torch.float)
        loss = model(img)
        loss = loss.data.cpu()

        for k in range(dmos.size(0)):
            # print(score_gt[k], final_score)
            results.append([dmos[k], loss])

            loss = 1000*nn.MSELoss()(dmos[k], loss)
            test_loss.append(loss)

    # results = np.array(results, dtype=float)
    results = torch.as_tensor(results, dtype=float)
    lcc = np.corrcoef(results, rowvar=False)[0][1]
    dmoss, preds = results[:, 0], results[:, 1]
    plcc = scipy.stats.pearsonr(dmoss, preds)[0]
    srocc = scipy.stats.spearmanr(dmoss, preds)[0]
    print('\nLCC: {}, PLCC: {}, SRCC: {}'.format(lcc, plcc, srocc))

    img = img.squeeze()
    return {
        'lcc': lcc,
        'plcc': plcc,
        'srocc': srocc,
        'test_loss': np.mean(test_loss),
        'pre_array': results[:, 0],
        'gt_array': results[:, 1],
        'img': img.data.cpu().numpy(),
    }
