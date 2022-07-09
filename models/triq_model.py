from backbone.resnet50 import ResNet50
from backbone.vgg16 import vgg16
from torchvision.models.resnet import resnet50

def create_triq_model(n_quality_levels,
                      backbone='resnet50',
                      transformer_params = (2, 32, 8, 64),
                      maximum_position_encoding=193,
                      vis=False):

    if backbone == 'resnet50':
        backbone_model = ResNet50(1000, channels=3)
    elif backbone == 'vgg16':
        backbone_model = vgg16(pretrained=False)
    else:
        raise NotImplementedError


