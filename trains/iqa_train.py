import torch

from torch import nn
from iqa_args import args
from trains.iqa_val import valid
import os


def snapshot(model, testloader, epoch, step, snapshot_dir, prefix=''):
    val_dict = valid(model, testloader)
    lcc = val_dict['lcc']
    srocc = val_dict['srocc']
    plcc = val_dict['plcc']
    test_loss = val_dict['test_loss']

    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args['output_dir'], "%s_checkpoint.bin" % args['name'])
    # torch.save(model_to_save.state_dict(), model_checkpoint)

    snapshot = {
        'epoch': epoch,
        'step': step,
        'model': model_to_save.state_dict(),
        'lcc': lcc,
        'srocc': srocc,
        'plcc': plcc
    }

    best = -1
    if lcc + srocc >= best:
        best = lcc + srocc
        torch.save(snapshot, os.path.join(snapshot_dir, '%s_%.4f_%.4f_epoch%d.pth' %
                                          (prefix, lcc, srocc, epoch)))

    torch.save(snapshot, os.path.join(snapshot_dir, '{0}.pth'.format(prefix)))

    print("[{}] Curr LCC: {:0.4f} SROCC: {:0.4f}, PLCC: {:0.4f}".format(epoch, lcc, srocc, plcc))

    out_dict = {'lcc': lcc,
                'srocc': srocc,
                'plcc': plcc,
                'best': best,
                'test_loss': test_loss,
                'pred': val_dict['pre_array'],
                'gt': val_dict['gt_array'],
                'img': val_dict['img'],
                }

    return out_dict


def trainProcess(model, optimizer, scheduler, t_total, trainloader, testloader, snapshot_dir, prefix=''):

    epochs = args['epoch']

    for epoch in range(1, epochs + 1):
        loss_score = []
        model.train()

        for step, (img, dmos) in enumerate(trainloader):


            img = torch.tensor(img, device='cuda', requires_grad=True, dtype=torch.float)
            dmos = torch.tensor(dmos, device='cuda', requires_grad=True, dtype=torch.float)
            # img = img.to('cuda')
            # dmos = dmos.to('cuda')


            loss = model(img)
            criterion = nn.MSELoss()
            loss = criterion(loss, dmos)
            loss = loss * 1000

            if args['gradient_accumulation_steps'] > 1:
                loss = loss / args['gradient_accumulation_steps']
            loss.backward()

            if (step + 1) % args['gradient_accumulation_steps'] == 0:
                # losses.up
                torch.nn.utils.clip_grad_norm(model.parameters(), args['max_grad_norm'])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if (step - 1) % t_total == 0:
                    break

            print("[{0}][{1}] Loss: {2:.4f}"
                  .format(epoch, step, loss))

            if step % 10 == 0:
                # model.train()
                # val_dict = valid(model, testloader)
                snapshot(model, testloader, epoch, step, snapshot_dir, prefix)

