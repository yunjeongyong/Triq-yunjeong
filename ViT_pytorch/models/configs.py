# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ml_collections




def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (32, 32)})
    config.hidden_size = 32
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 64
    config.transformer.num_heads = 8
    config.transformer.num_layers = 2
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    config.resnet_pretrained_path = None
    config.pretrained_path = './ViT-B_16.npz'
    return config


def get_r50_b16_config():
    """Returns the Resnet50 + ViT-B/16 configuration."""
    config = get_b16_config()
    del config.patches.size
    config.patches.grid = (14, 14)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1
    # config.resnet.width_factor = 0.5
    return config




# def triq_config():
#     confi_g = {
#         "num_layers": 2,
#         "d_model": 32,
#         "num_heads": 8,
#         "mlp_dim": 64,
#         "dropout": 0.1,
#         "n_quality_levels": 5,
#         "maximum_position_encoding": 257,
#         "vis": False,
#     }
#     return confi_g


