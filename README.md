# blenderbot_paddle
用Paddle复现[Recipes for building an open-domain chatbot](https://aclanthology.org/2021.eacl-main.24.pdf)论文

论文名称： [Recipes for building an open-domain chatbot](https://aclanthology.org/2021.eacl-main.24.pdf)
数据集： 无
验收标准： 

1. 复现Blenderbot和Blenderbot Small两个模型（参考论文和实现链接）

2. 完成模型权重从pytorch到paddle的转换代码，
   转换4个预训练权重（90M，2.7B，2.7B distilled to 1.4B，2.7B distilled to 360M）
3. Blenderbot For Conditional Generation模型和Blenderbot Small For Conditional Generation
   模型前向推理输出对齐参考代码（90M和2.7B distilled to 360M两个权重）
4. 提交PR至PaddleNLP

### tokenizer核对

核对 blenderbotsmall 的tokenizer

```
python tokenizer_check.py --model_name=blenderbot_small-90M
```

> input text: My friends are cool but they eat too many carbs.
> torch tokenizer:  [42, 643, 46, 1430, 45, 52, 1176, 146, 177, 753, 2430, 5]
> paddle tokenizer:  [42, 643, 46, 1430, 45, 52, 1176, 146, 177, 753, 2430, 5]
> input text: My 'but' they:@ eat too many carbs:)
> torch tokenizer:  [42, 8, 45, 8, 2277, 332, 3, 6708, 1176, 146, 177, 753, 372, 330, 106, 39]
> paddle tokenizer:  [42, 8, 45, 8, 2277, 332, 3, 6708, 1176, 146, 177, 753, 372, 330, 106, 39]

核对 blenderbot 的 tokenizer

```
python tokenizer_check.py --model_name=blenderbot-400M-distill
```

> input text: My friends are cool but they eat too many carbs.
> torch tokenizer:  [863, 1329, 366, 1449, 373, 382, 1861, 618, 847, 911, 1372, 21, 2]
> paddle tokenizer:  [863, 1329, 366, 1449, 373, 382, 1861, 618, 847, 911, 1372, 21, 2]
> input text: My 'but' they:@ eat too many carbs:)
> torch tokenizer:  [863, 1069, 2871, 14, 382, 33, 39, 1861, 618, 847, 911, 1372, 33, 16, 2]
> paddle tokenizer:  [863, 1069, 2871, 14, 382, 33, 39, 1861, 618, 847, 911, 1372, 33, 16, 2]

#### 权重转换

```python
. ./convert.sh
```

#### 精度校验

```shell
python model_check.py --model_name=blenderbot-400M-distill
```

```
huggingface facebook/blenderbot-400M-distill vs paddle blenderbot-400M-distill
input text: My friends are cool but they eat too many carbs.
mean difference: tensor(8.6926e-07)
max difference: tensor(9.5367e-06)
```

```shell
python model_check.py --model_name=blenderbot_small-90M
```

```
huggingface facebook/blenderbot_small-90M vs paddle blenderbot_small-90M
input text: My friends are cool but they eat too many carbs.
mean difference: tensor(1.7785e-06)
max difference: tensor(1.0014e-05)
```

#### 注重点

|                                     | small-90M | normal |
| ----------------------------------- | --------- | ------ |
| Normalize_before                    | False     | True   |
| encoder/decoder layer norm position | before    | after  |
| config与词表区别大                  |           |        |


