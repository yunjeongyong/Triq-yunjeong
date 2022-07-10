import numpy as np
import torch
import torch.optim as optim
from models.transformer_iqa_2 import VisionTransformer, CONFIGS
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torchvision import transforms
import ViT_pytorch.models.configs as configs
from iqa_dataloader import KONIDataset
from iqa_train import trainProcess
from iqa_args import args


def set_model():
    config = configs.get_r50_b16_config()
    model = VisionTransformer(config, zero_head=True, load_transformer_weights=True)
    model.load_from(np.load(args['pretrained_dir']))
    model = torch.nn.DataParallel(model)
    return model


def set_optimizer(model):
    # optimG = optim.SGD(filter(lambda p: p.requires_grad, \
    #     model.parameters()),lr=args.lr,momentum=0.9,\
    #     weight_decay=1e-4,nesterov=True)
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=args['learning_rate'],
                                weight_decay=args['weight_decay'])
    # TODO: add scheduler
    scheduler = None

    return optimizer, scheduler


def main():
    csv_path = args['csv_path']
    data_path = args['data_path']
    train_batch_size = args['train_batch_size']
    test_batch_size = args['test_batch_size']

    trainset = KONIDataset(csv_path, data_path, is_train=True)
    testset = KONIDataset(csv_path, data_path, is_train=False)

    trainloader = DataLoader(trainset,
                             batch_size=train_batch_size,
                             shuffle=True,
                             num_workers=2,
                             drop_last=True,
                             pin_memory=True)
    testloader = DataLoader(testset,
                            batch_size=test_batch_size,
                            shuffle=True,
                            num_workers=1,
                            pin_memory=True)

    model = set_model()
    optimizer, scheduler = set_optimizer(model=model)

    trainProcess(
        model,
        optimizer,
        scheduler,
        trainloader,
        testloader,
        args['snapshot_dir']
    )


if __name__ == "__main__":
    main()
