# blenderbot_paddle

用Paddle复现[Recipes for building an open-domain chatbot](https://aclanthology.org/2021.eacl-main.24.pdf)论文

开放式的聊天系统一直是机械学习/深度学习领域的一个巨大挑战。Blenderbot一文展示了大规模模型在对训练数据和对话生成方式进行合理选择后，可以在对话中做到强调重点，保持对话个性与聊天基调等。

本次复现的模型为 Blenderbot（对应论文中2.7B模型 ） 与 Blenderbot small （对应论文中90M模型)）

### tokenizer核对

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

```shell
python model_check.py --model_name=blenderbot-400M-distill
```

![image-20210809182542119](img/README/image-20210809182542119.png)

```shell
python model_check.py --model_name=blenderbot_small-90M
```

![image-20210810120030476](img/README/image-20210810120030476.png)

对官方给出的例句与随意例句，前向传导后的logits误差都在1E-5级别。

#### 注重点

|                                     | small-90M | normal |
| ----------------------------------- | --------- | ------ |
| Normalize_before                    | False     | True   |
| encoder/decoder layer norm position | before    | after  |
| config与词表区别大                  |           |        |

