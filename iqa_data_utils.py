import numpy as np
import torch
from models.transformer_iqa_2 import VisionTransformer
from torch.utils.data import DataLoader
import ViT_pytorch.models.configs as configs
from iqa_dataloader import KONIDataset
from trains.iqa_train import trainProcess
from iqa_args import args
from ViT_pytorch.utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule

args['device'] = torch.device("cuda:%s" % args['GPU_ID'] if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('Using GPU %s' % args['GPU_ID'])
else:
    print('Using CPU')

def set_model():
    config = configs.get_r50_b16_config()
    # model = VisionTransformer(config, zero_head=True, load_transformer_weights=True)
    model_transformer = VisionTransformer(config, zero_head=True, load_transformer_weights=True).to(args['device'])
    # model = torch.nn.DataParallel(model)
    return model_transformer


def set_optimizer(model, trainloader):
    # optimG = optim.SGD(filter(lambda p: p.requires_grad, \
    #     model.parameters()),lr=args.lr,momentum=0.9,\
    #     weight_decay=1e-4,nesterov=True)
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=args['learning_rate'],
                                weight_decay=args['weight_decay'])
    # TODO: add scheduler
    train_steps = trainloader.__len__()
    num_warmup_steps = args['warmup_steps'] * train_steps
    t_total = args['epoch'] * train_steps
    if args['decay_type'] == 'cosine':
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=t_total)
    return optimizer, scheduler, t_total


def main():
    csv_path = args['csv_path']
    data_path = args['data_path']
    train_batch_size = args['train_batch_size']
    test_batch_size = args['test_batch_size']

    print('csv_path: {0}, data_path: {1}, train_batch_size: {2}, test_batch_size: {3}'
          .format(csv_path, data_path, train_batch_size, test_batch_size))



    trainset = KONIDataset(csv_path, data_path, img_size=args['img_size'], is_train=True)
    testset = KONIDataset(csv_path, data_path, img_size=args['img_size'], is_train=False)

    trainloader = DataLoader(trainset,
                             batch_size=train_batch_size,
                             shuffle=True,
                             num_workers=0,
                             drop_last=True,
                             pin_memory=False)
    testloader = DataLoader(testset,
                            batch_size=test_batch_size,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=False)

    model = set_model()
    optimizer, scheduler, t_total = set_optimizer(model=model, trainloader=trainloader)

    trainProcess(
        model,

        optimizer,
        scheduler,
        t_total,
        trainloader,
        testloader,
        args['snapshot_dir'],
        prefix=''
    )


if __name__ == "__main__":
    main()
