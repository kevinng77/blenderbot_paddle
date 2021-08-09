# blenderbot_paddle

用Paddle复现[Recipes for building an open-domain chatbot](https://aclanthology.org/2021.eacl-main.24.pdf)论文

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

```python
 python convert.py --model_name=blenderbot-400M-distill --torch_file_folder=../../../下载
```

程序会从 `--torch_file_folder/model_name/pytorch_model.bin` 加载torch 权重，以上面代码为例，加载路径为 `../../../下载/blenderbot-400M-distill/pytorch_model.bin`

默认输出路径为 `./blenderbot-400M-distill/model_state.pdparams`

#### 精度校验

```shell
python model_check.py --model_name=blenderbot-400M-distill
```

![image-20210809182542119](img/README/image-20210809182542119.png)

```shell
python model_check.py --model_name=blenderbot_small-90M
```

![image-20210809182647118](img/README/image-20210809182647118.png)

对官方给出的例句与随意例句，前向传导后的logits误差都在1E-5级别。

#### 注重点

|                                     | small-90M | normal |
| ----------------------------------- | --------- | ------ |
| Normalize_before                    | False     | True   |
| encoder/decoder layer norm position | before    | after  |
| config与词表区别大                  |           |        |

