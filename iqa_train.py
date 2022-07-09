import torch

from iqa_data_utils import main
# from iqa_data_utils import args
from iqa_args import args


def trainProcess(model, optimizer, trainloader, testloader, max_epoch, snapshot_dir, prefix, is_first):
    best_lcc = -1

    for epoch in range(1, max_epoch + 1):
        loss_score = []
        model.train()

        for step, (img, dmos) in enumerate(trainloader):

            img = torch.tensor(img, device='cuda', requires_grad=True, dtype=torch.float)
            dmos = torch.tensor(dmos, device='cuda', requires_grad=True, dtype=torch.float)

            loss = model(img, dmos)

            if args['gradient_accumulation_steps'] > 1:
                loss = loss / args['gradient_accumulation_steps']
            loss.sum().backward()

            if (step + 1) % args['gradient_accumulation_steps'] == 0:
                losses.up
            optimizer.zero_grad()







