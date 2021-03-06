# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict
import argparse
import re
import torch
import paddle


def convert_pytorch_checkpoint_to_paddle(pytorch_checkpoint_path,
                                         paddle_dump_path):
    pytorch_state_dict = torch.load(pytorch_checkpoint_path, map_location="cpu")
    paddle_state_dict = OrderedDict()
    paddle.set_device("cpu")

    for k, v in pytorch_state_dict.items():
        if k in skip_weights:
            continue
        if k[-7:] == ".weight":
            if not any([w in k for w in dont_transpose]):
                if v.ndim == 2:
                    v = v.transpose(0, 1)

        if k.startswith('model.decoder.'):
            mapping = decoder_mapping
            tw_offset = "model.decoder."
            pw_offset = f"{model}.decoder."
        elif k.startswith('model.encoder'):
            mapping = encoder_mapping
            tw_offset = "model.encoder."
            pw_offset = f"{model}.encoder."
        else:
            mapping = other_maps
            tw_offset = ""
            pw_offset = ""
        for huggingface_name, paddle_name in mapping.items():
            k = re.sub(tw_offset + huggingface_name, pw_offset + paddle_name, k)
        paddle_state_dict[k] = v.data.numpy().astype(args.dtype)
    paddle.save(paddle_state_dict, paddle_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="blenderbot-400M-distill")
    parser.add_argument("--torch_file_folder", type=str, default="../../??????")
    parser.add_argument('--dtype',type=str,default='float32',help="float32 or float16")

    args = parser.parse_args()
    assert args.dtype in ['float32','float16'],\
        f"{args.dtype} dtype only support float32 or float16"

    pytorch_checkpoint_path = f"{args.torch_file_folder}/{args.model_name}/pytorch_model.bin"
    paddle_dump_path = f"./data/{args.model_name}/model_state.pdparams"

    if args.model_name == "blenderbot_small-90M":
        model = "blenderbot_small"
    elif args.model_name in ["blenderbot-400M-distill",
                             "blenderbot-1B-distill",
                             "blenderbot-3B"]:
        model = "blenderbot"
    else:
        raise f"{args.model_name} not find"
    print('converting ', args.model_name)
    print('loading from ', pytorch_checkpoint_path)
    print('output path ', paddle_dump_path)

    decoder_mapping = {
        "embed_positions": "decoder_embed_positions",
        "layernorm_embedding": "decoder_layernorm_embedding",
        r"layers.(\d+).encoder_attn.(\w+)_proj": r"decoder.layers.\1.cross_attn.\2_proj",
        r"layers.(\d+).fc(\d)": r"decoder.layers.\1.linear\2",
        r"layers.(\d+).self_attn.(\w+)_proj": r"decoder.layers.\1.self_attn.\2_proj",
        r"layers.(\d+).self_attn_layer_norm": r"decoder.layers.\1.norm1",
        r"layers.(\d+).final_layer_norm": r"decoder.layers.\1.norm3",
        r"layers.(\d+).encoder_attn_layer_norm": r"decoder.layers.\1.norm2",
        "embed_tokens": "embed_tokens",
        "layer_norm": "decoder_layernorm"
    }

    encoder_mapping = {
        "embed_positions": "encoder_embed_positions",
        "layernorm_embedding": "encoder_layernorm_embedding",
        r"layers.(\d+).decoder_attn.(\w+)_proj": r"encoder.layers.\1.cross_attn.\2_proj",
        r"layers.(\d+).fc(\d)": r"encoder.layers.\1.linear\2",
        r"layers.(\d+).self_attn.(\w+)_proj": r"encoder.layers.\1.self_attn.\2_proj",
        r"layers.(\d+).self_attn_layer_norm": r"encoder.layers.\1.norm1",
        r"layers.(\d+).final_layer_norm": r"encoder.layers.\1.norm2",
        r"layers.(\d+).decoder_attn_layer_norm": r"encoder.layers.\1.norm3",
        "embed_tokens": "embed_tokens",
        "layer_norm": "encoder_layernorm"
    }

    other_maps = {
        "final_logits_bias": "final_logits_bias",
        "lm_head.weight": "lm_head_weight",
        "model.shared.weight": f"{model}.shared.weight",
    }

    skip_weights = []
    dont_transpose = [
        "_embeddings.weight","_layer_norm",
        "embed_positions", "embed_tokens", "layernorm_embedding",
        "lm_head", "shared"
    ]

    convert_pytorch_checkpoint_to_paddle(pytorch_checkpoint_path,paddle_dump_path)
