# blenderbot_paddle

用Paddle复现[Recipes for building an open-domain chatbot](https://aclanthology.org/2021.eacl-main.24.pdf)论文

开放式的聊天系统一直是机械学习/深度学习领域的一个巨大挑战。Blenderbot一文展示了大规模模型在对训练数据和对话生成方式进行合理选择后，可以在对话中做到强调重点，保持对话个性与聊天基调等。

本次复现的模型为 Blenderbot（对应论文中2.7B模型 ） 与 Blenderbot small （对应论文中90M模型)）

#### 环境依赖

```
pip install -r requirements.txt
```

如果要进行权重转换及模型前向传导测试，还需安装torch与transformers。本次使用的依赖版本如下

```python
torch=1.7.1
transformers=4.9.1
paddlepaddle=2.1.2
```

#### tokenizer核对

> 本仓库实现了 tokenizer与transformers 的对齐

核对 blenderbotsmall 的tokenizer

```
python tokenizer_check.py --model_name=blenderbot_small-90M
```

> input text: My friends are cool but they eat too many carbs.
>
> torch tokenizer:  [42, 643, 46, 1430, 45, 52, 1176, 146, 177, 753, 2430, 5]
>
> paddle tokenizer:  [42, 643, 46, 1430, 45, 52, 1176, 146, 177, 753, 2430, 5]
>
> input text: My 'but' they:@ eat too many carbs:)
>
> torch tokenizer:  [42, 8, 45, 8, 2277, 332, 3, 6708, 1176, 146, 177, 753, 372, 330, 106, 39]
>
> paddle tokenizer:  [42, 8, 45, 8, 2277, 332, 3, 6708, 1176, 146, 177, 753, 372, 330, 106, 39]

核对 blenderbot 的 tokenizer

```
python tokenizer_check.py --model_name=blenderbot-400M-distill
```

> input text: My friends are cool but they eat too many carbs.
>
> torch tokenizer:  [863, 1329, 366, 1449, 373, 382, 1861, 618, 847, 911, 1372, 21, 2]
>
> paddle tokenizer:  [863, 1329, 366, 1449, 373, 382, 1861, 618, 847, 911, 1372, 21, 2]
>
> input text: My 'but' they:@ eat too many carbs:)
>
> torch tokenizer:  [863, 1069, 2871, 14, 382, 33, 39, 1861, 618, 847, 911, 1372, 33, 16, 2]
>
> paddle tokenizer:  [863, 1069, 2871, 14, 382, 33, 39, 1861, 618, 847, 911, 1372, 33, 16, 2]

#### 权重转换

将 [Hugging Face](https://huggingface.co/models?search=blender) 上的 blenderbot-400M-distill, blenderbot_small-90M, blenderbot-1B-distill, blenderbot-3B 四个模型进行转换。转换前需要模型到对应的目录下。

```python
 python convert.py --model_name=blenderbot-400M-distill --torch_file_folder=../../../下载
```

程序会从 `--torch_file_folder/model_name/pytorch_model.bin` 加载torch 权重，以上面代码为例，加载路径为 `../../../下载/blenderbot-400M-distill/pytorch_model.bin`

默认输出路径为 `./blenderbot-400M-distill/model_state.pdparams`

转换后的paddle 权重下载链接：

链接: https://pan.baidu.com/s/1MGHSE4Q_mXEMuYT3CwzJiA  密码: lgl5

#### 精度校验

官方要求的 `blenderbot-400M-distill` 与 `blenderbot_small-90M` 模型校验：

```shell
python model_check.py --model_name=blenderbot-400M-distill
```

![image-20210809182542119](img/README/image-20210809182542119.png)

```shell
python model_check.py --model_name=blenderbot_small-90M
```

![image-20210810120030476](img/README/image-20210810120030476.png)

对 transformers 上给出的例句与随意例句，前向传导后的logits误差都在1E-5级别。

其他两个模型的前向传导检验:

`blenderbot-1B-distill`

![image-20210810125823870](img/README/image-20210810125823870.png)

`blenderbot-3B ` 的权重是在太大了，在个人电脑上跑不动，因此也就没有做前向传导的对比测试了。

#### 两个模型的对比注重点

| Hugging face 中的 config 不同 | small-90M | normal |
| ----------------------------- | --------- | ------ |
| Normalize_before              | False     | True   |
| add_final_layer_norm          | False     | True   |
| normalize_embedding           | True      | False  |

+ `normalize_before` 对应 `nn.TransformerEncoderLayer` 的 `normalize_before` 参数
+ `normalize_embedding` True 时， enocder 与 decoder 在执行完 token 到 embedding的转换后，会对`input_embeds` 进行 layer norm。可参考blenderbot small modeling代码中的238行左右。
+ `add_final_layer_norm` 为True时，会在encoder 与 decoder结束后，会对encoder_output/decoder_output 进行layer norm。具体可参考 blenderbot 220行左右代码。

**hugging face config 文件中，生成文本的相关配置没有加载进来，如：**

```python
{"length_penalty": 0.65,
  "max_length": 60,
  "min_length": 20,
  "num_beams": 10,
  "force_bos_token_to_be_generated": false,
  "forced_eos_token_id": 2,}
```

以下这些 config 文件中的参数没有在 paddle 中设置：

```python
{
  "classif_dropout": 0.0,
  "decoder_layerdrop": 0.0,
  "encoder_layerdrop": 0.0,
  "do_blenderbot_90_layernorm": true, 
  "static_position_embeddings": false,
  "use_cache": true,
  "num_hidden_layers": 2,
  "layernorm_variant": "prelayernorm",
  "is_encoder_decoder": true,
  "encoder_no_repeat_ngram_size": 3,
}
```

对于： `do_lenderbot_90_layernorm` 一项，在transformers 的 [configuration文件](https://huggingface.co/transformers/v3.4.0/_modules/transformers/configuration_blenderbot.html) 中有提及，但在 transformers.model.blenderbot 中并没有找到相关应用。 部分网友有对这个参数进行[描述](https://gist.github.com/sshleifer/cb245b8739420724a32fc0c22344aee0) 但并没有在transformers中对应上，不知道是不是版本问题导致的。

Blenderbot与BlenderbotSmall在Transformers 中的 `use_cache` 参数为 True，与paddle `TransformersDecoder`/ `TransformersEncoder` 的默认计算方法一致。

decoder_layerdrop，encoder_layerdrop 两个模型均为0，因为 paddle 中的  `TransformersDecoder`/ `TransformersEncoder` 似乎没有layer drop的设置,（不太确定，因为个人没找到相关layer drop的设置）所以此时不传递这个参数并没有影响。

`classif_dropout` 在  transformers.model.blenderbot中也是没有找到相关的应用
