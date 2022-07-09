import copy
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from ml_collections.config_flags.examples.define_config_dict_basic import config
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm

from ViT_pytorch.models import configgs
from ViT_pytorch.models.modeling_resnet import ResNetV2
from ViT_pytorch.models.configs import get_r50_b16_config
import logging

logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

def create_padding_mask(input):

    p1d = [0, 0, 1, 0, 0, 0]
    input = F.pad(input, p1d, "constant", 1)
    input = torch.equal(torch.mean(input, dim=-1), torch.zeros(input.size()))
    input = input.type(torch.FloatTensor)

    return input[:, 1, 1, :]

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)

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


class MultiHeadAttention(nn.Module):
    def __init__(self, confi_g, num_heads = 8):
        super(MultiHeadAttention, self).__init__()
        self.confi_g = confi_g
        self.num_heads = confi_g["num_heads"]
        self.d_model = confi_g["d_model"]
        self.depth = self.d_model // self.num_heads
        self.all_head_size = self.num_heads * self.depth
        self.dropout = confi_g["dropout"]

        # all = n_head * depth
        self.query = nn.Linear(self.d_model, self.all_head_size)
        self.key = nn.Linear(self.d_model, self.all_head_size)
        self.value = Linear(self.d_model, self.all_head_size)

        self.out = Linear(self.d_model, self.d_model)
        self.attn_dropout = Dropout(self.dropout)
        self.proj_dropout = Dropout(self.dropout)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def scaled_dot_product_attention(self, query, key, value, mask):
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if mask is not None:
            attention_scores += (mask * -1e9)
        attention_probs = self.softmax(attention_scores)
        output = torch.matmul(attention_probs, value)
        return output, attention_probs

    def forward(self, inputs, mask):
        mixed_query_layer = self.query(inputs)
        mixed_key_layer = self.key(inputs)
        mixed_value_layer = self.value(inputs)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention, weights = self.scaled_dot_product_attention(query_layer, key_layer, value_layer, mask)
        attention = attention.permute(0, 2, 1, 3)
        new_attention = attention.size()[:-2] + (self.all_head_size,)
        attention = attention.view(*new_attention)
        attention_output = self.out(attention)
        attention_output = self.proj_dropout(attention_output)

        return attention_output, weights


class MLP(nn.Module):
    def __init__(self, confi_g):
        super(MLP, self).__init__()
        self.d_model = confi_g["d_model"]
        self.mlp_dim = confi_g["mlp_dim"]
        self.fc1 = nn.Linear(self.d_model, self.mlp_dim)
        self.fc2 = nn.Linear(self.mlp_dim, self.d_model)
        self.act_fn = F.relu

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)

        return x

class MLP_gelu(nn.Module):
    def __init__(self, confi_g, mlp_activation):
        super(MLP_gelu, self).__init__()
        self.d_model = confi_g["d_model"]
        self.mlp_dim = confi_g["mlp_dim"]
        self.fc1 = nn.Linear(self.d_model, self.mlp_dim)
        self.fc2 = nn.Linear(self.mlp_dim, confi_g["n_quality_levels"])
        self.dropout = nn.Dropout(confi_g["dropout_rate"])
        self.act_fn = F.gelu
        self.act_fn2 = mlp_activation

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.act_fn2(x)

        return x
class TransformerBlock(nn.Module):
    def __init__(self, confi_g):
        super(TransformerBlock, self).__init__()
        self.d_model = confi_g["d_model"]
        self.mlp_dim = confi_g["mlp_dim"]
        self.dropout = confi_g["dropout"]
        self.vis = confi_g["vis"]

        self.mha = MultiHeadAttention(confi_g)
        self.ffn = MLP(confi_g)

        self.layernorm1 = nn.LayerNorm(self.d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(self.d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)

    def forward(self, x, training, mask, confi_g):
        h = x
        x, weights = self.mha(x, mask)
        x = self.dropout1(x)
        out1 = self.layernorm1(x + h)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)

        if self.vis:
            return out2, weights
        else:
            return out2

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[ROOT + "/" + ATTENTION_Q + "/" + "kernel"]).view(self.hidden_size,
                                                                                          self.hidden_size).t()
            key_weight = np2th(weights[ROOT + "/" + ATTENTION_Q + "/" + "kernel"]).view(self.hidden_size,
                                                                                        self.hidden_size).t()
            value_weight = np2th(weights[ROOT + "/" + ATTENTION_Q + "/" + "kernel"]).view(self.hidden_size,
                                                                                          self.hidden_size).t()
            out_weight = np2th(weights[ROOT + "/" + ATTENTION_Q + "/" + "kernel"]).view(self.hidden_size,
                                                                                        self.hidden_size).t()

            query_bias = np2th(weights[ROOT + "/" + ATTENTION_Q + "/" + "bias"]).view(-1)
            key_bias = np2th(weights[ROOT + "/" + ATTENTION_Q + "/" + "bias"]).view(-1)
            value_bias = np2th(weights[ROOT + "/" + ATTENTION_Q + "/" + "bias"]).view(-1)
            out_bias = np2th(weights[ROOT + "/" + ATTENTION_Q + "/" + "bias"]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[ROOT + "/" + FC_0 + "/" + "kernel"]).t()
            mlp_weight_1 = np2th(weights[ROOT + "/" + FC_1 + "/" + "kernel"]).t()
            mlp_bias_0 = np2th(weights[ROOT + "/" + FC_0 + "/" + "bias"]).t()
            mlp_bias_1 = np2th(weights[ROOT + "/" + FC_1 + "/" + "bias"]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[ROOT + "/" + ATTENTION_NORM + "/" + "scale"]))
            self.attention_norm.bias.copy_(np2th(weights[ROOT + "/" + ATTENTION_NORM + "/" + "bias"]))
            self.ffn_norm.weight.copy_(np2th(weights[ROOT + "/" + MLP_NORM + "/" + "scale"]))
            self.ffn_norm.bias.copy_(np2th(weights[ROOT + "/" + MLP_NORM + "/" + "bias"]))

class Encoder(nn.Module):
    def __init__(self, confi_g):
        super(Encoder, self).__init__()
        self.vis = confi_g["vis"]
        self.layer = nn.ModuleList()
        for _ in range(confi_g["num_layers"]):
            layer = TransformerBlock(confi_g)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x):
        if self.vis:
            attn_weights = []
            for layer_block in self.layer:
                x, weights = layer_block(x)
                attn_weights.append(weights)
            return x, attn_weights
        else:
            for layer_block in self.layer:
                x = layer_block(x)
            return x


class Transformer(nn.Module):
    def __init__(self, confi_g):
        super(Transformer, self).__init__()
        self.vis = confi_g["vis"]
        self.embeddings = TriQImageQualityTransformer(confi_g, config)
        self.encoder = Encoder(confi_g)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        if self.vis:
            encoded, attn_weights = self.encoder(embedding_output)
            return encoded, attn_weights
        else:
            encoded = self.encoder(embedding_output)
            return encoded




class TriQImageQualityTransformer(nn.Module):
    def __init__(self, confi_g, config, in_channels=3):
        super(TriQImageQualityTransformer, self).__init__()

        self.d_model = confi_g["d_model"]
        self.num_layers = confi_g["num_layers"]
        self.pos_emb = nn.Parameter(torch.zeros(1, confi_g["maximum_position_encoding"], self.d_model))
        self.quality_emb = nn.Parameter(torch.zeros(1, 1, self.d_model))

        self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
                                     width_factor=config.resnet.width_factor)
        in_channels = self.hybrid_model.width * 16
        self.feature_proj_conv = Conv2d(in_channels=in_channels,
                                        out_channels=self.d_model,
                                        kernel_size=1,
                                        stride=1)
        self.pooling_small = nn.MaxPool2d(kernel_size=2)
        self.dropout = Dropout(confi_g["dropout"])
        self.enc_layers = Encoder(confi_g)

        if confi_g["n_quality_levels"] > 1:
            # mlp_activation = 'softmax'
            mlp_activation = nn.Softmax
        else:
            # mlp_activation = 'linear'
            mlp_activation = nn.Linear
        self.mlp = MLP_gelu(confi_g, mlp_activation)

    def forward(self, x, training):
        B = x.shape[0]
        mask = None
        x = self.feature_proj_conv(x)

        if x.shape[1] >= 16:
            x = self.pooling_small(x)

        spatial_size = x.shape[1] * x.shape[2]
        x = torch.reshape(x, (B, spatial_size, self.d_model))

        quality_emb = torch.broadcast_tensors(self.quality_emb, (B, 1, self.d_model))
        x = torch.cat((quality_emb, x), dim=1)

        x = x + self.pos_emb[:, x.shape[1], :]
        x = self.dropout(x)

        x = self.mlp(x[:, 0])
        if self.vis:
            x, attn_weights = self.enc_layers(x, training, mask)
            return x, attn_weights
        else:
            x = self.enc_layers(x, training, mask)
            return x

CONFIGS = {
    'ViT-B_16': configgs.get_b16_config(),
    'ViT-B_32': configgs.get_b32_config(),
    'ViT-L_16': configgs.get_l16_config(),
    'ViT-L_32': configgs.get_l32_config(),
    'ViT-H_14': configgs.get_h14_config(),
    'R50-ViT-B_16': configgs.get_r50_b16_config(),
    'R50-ViT-Simple': configgs.get_r50_simple_config(),
    'testing': configgs.get_testing(),
}



if __name__ == '__main__':
    # def triq_config():
    #     config = {
    #         "num_layers": 2,
    #         "d_model": 32,
    #         "num_heads": 8,
    #         "mlp_dim": 64,
    #         "dropout": 0.1,
    #         "n_quality_levels": 5,
    #         "maximum_position_encoding": 257,
    #         "vis": False,
    #     }
    #     return config
    confi_g = triq_config()

    encoder = ResNetV2((3, 4, 9), 1)
    projection = nn.Conv2d(1024, 32, 1, 1)
    maxpool = nn.MaxPool2d(2, 2)


    in_t = torch.rand(8, 3, 224, 224)
    code = encoder(in_t)
    proj = projection(code)
    seq = maxpool(proj)
    token_seq = seq.view(8, -1, 32)
    print(token_seq.shape)

    # model = Transformer(config)
    # out = model(in_t)
    # model = TriQImageQualityTransformer(confi_g, config, in_channels=3)
    # out = model(in_t)




