# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
from iqa_args import args
from os.path import join as pjoin
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

import ViT_pytorch.models.configs as configs

from ViT_pytorch.models.modeling_resnet import ResNetV2
from backbone.resnet50 import resnet50


logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

args['device'] = torch.device("cuda:%s" % args['GPU_ID'] if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('Using GPU %s' % args['GPU_ID'])
else:
    print('Using CPU')

def triq_config():
    confi_g = {
        "num_layers": 2,
        "d_model": 32,
        "num_heads": 8,
        "mlp_dim": 64,
        "dropout": 0.1,
        "n_quality_levels": 5,
        "maximum_position_encoding": 257,
        "vis": False,
    }
    return confi_g

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        # num_heads(num_attention_heads): 8
        # hidden_size: 32
        # attention_head_size: 4
        # all_head_size: 32
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, in_channels=2048, n_patches=192):
        super(Embeddings, self).__init__()
        self.hybrid = None

        # if config.patches.get("grid") is not None:
        #     grid_size = config.patches["grid"]
        #     patch_size = (1, 1)
        #     n_patches = n_patches
        #     self.hybrid = True
        # else:
        #     patch_size = _pair(config.patches["size"])
        #     n_patches = n_patches
        #     self.hybrid = False

        # self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
        #                                  width_factor=config.resnet.width_factor)
        self.hybrid_model = resnet50(pretrained=True).to(args['device'])
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=1,
                                       stride=1)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])
        self.pooling_small = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        # if x.shape[-1] >= 64:
        #     x = self.pooling_big(x)
        # elif x.shape[-1] >= 24:
        #     x = self.pooling_small(x)

        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings[:, x.shape[1], :]
        # embeddings = self.dropout(embeddings)
        return embeddings


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights




class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        print(input_ids.shape)
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


def smooth_Label_cross_entropy(input, target):
    logprobs = torch.nn.functional.log_softmax(input, dim = 1)
    return  -(target * logprobs).sum() / input.shape[0]


def categorical_cross_entropy(y_pred, y_true):
    y_pred = torch.clamp(y_pred, 1e-9, 1 - 1e-9)
    return -(y_true * torch.log(y_pred)).sum(dim=1).mean()


class VisionTransformer(nn.Module):
    def __init__(self, config, zero_head=False, vis=False, load_transformer_weights=True):
        super(VisionTransformer, self).__init__()
        self.zero_head = zero_head

        self.transformer = Transformer(config, vis)
        self.head = Linear(config.hidden_size, 1)
        self.load_transformer_weights = load_transformer_weights

    def forward(self, x):
        x, attn_weights = self.transformer(x)
        output = self.head(x[:, 0])
        # logits = nn.Softmax(logits)

        return output

#     def load_from(self, weights):
#         with torch.no_grad():
#             if self.load_transformer_weights:
#                 if self.zero_head:
#                     nn.init.zeros_(self.head.weight)
#                     nn.init.zeros_(self.head.bias)
#                 else:
#                     self.head.weight.copy_(np2th(weights["head/kernel"]).t())
#                     self.head.bias.copy_(np2th(weights["head/bias"]).t())
#
#                 temp = weights["embedding/kernel"]
#                 print(temp.shape)
#                 self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
#                 self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
#                 self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
#                 self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
#                 self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))
#
#                 posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
#                 posemb_new = self.transformer.embeddings.position_embeddings
#                 if posemb.size() == posemb_new.size():
#                     self.transformer.embeddings.position_embeddings.copy_(posemb)
#                 else:
#                     logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
#                     ntok_new = posemb_new.size(1)
#
#                     # if self.classifier == "token":
#                     #     posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
#                     #     ntok_new -= 1
#                     # else:
#                     #     posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
#                     posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
#
#                     gs_old = int(np.sqrt(len(posemb_grid)))
#                     gs_new = int(np.sqrt(ntok_new))
#                     print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
#                     posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
#
#                     zoom = (gs_new / gs_old, gs_new / gs_old, 1)
#                     posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
#                     posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
#                     posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
#                     self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))
#
#                 for bname, block in self.transformer.encoder.named_children():
#                     for uname, unit in block.named_children():
#                         unit.load_from(weights, n_block=uname)
#
#             if self.transformer.embeddings.hybrid:
#                 self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
#                 gn_weight = np2th(weights["gn_root/scale"]).view(-1)
#                 gn_bias = np2th(weights["gn_root/bias"]).view(-1)
#                 self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
#                 self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)
#
#                 for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
#                     for uname, unit in block.named_children():
#                         unit.load_from(weights, n_block=bname, n_unit=uname)
#
# class IQARegression(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#
#         self.conv_enc = nn.Conv2d(in_channels=2048, out_channels=config.hidden_size, kernel_size=1, stride=1)





CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
}


if __name__ == '__main__':

    # config = CONFIGS['R50-ViT-B_16']
    # in_t = torch.rand(8, 3, 224, 224)
    # model = VisionTransformer(config, zero_head=True, num_classes = 5)
    # out = model(in_t)
    # encoder = ResNetV2((3, 4, 9), 1)
    encoder = resnet50(1, 3)
    #
    # print(encoder)
    projection = nn.Conv2d(2048, 32, 1, 1)
    maxpool = nn.MaxPool2d(2, 2)

    in_t = torch.rand(8, 3, 512, 384)
    code = encoder(in_t)
    print(code.shape)
    proj = projection(code)
    seq = maxpool(proj)
    print(seq.shape)
    token_seq = seq.flatten(2)
    x = token_seq.transpose(-1, -2)
    print(x.shape)