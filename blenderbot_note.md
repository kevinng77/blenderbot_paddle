## PR review

- [x] Add an additional line behind since we are referencing the transformers repo.

  ` # Copyright 2021 The Facebook, Inc. and The HuggingFace Inc. team.`

- [x] It seems you only changed the default value of arg return_attention_mask. It might be unnecessary to override the **call** method. 

+ 移除了这个 call method

- [x] Please add docstrings for all your classes and methods which might be utilized by users. You can refer to paddlenlp.transformers.bert. We use Google Style docstrings. Refer to Bert Model for reference. 
- [x] Please also included the public tokenize method. 

+ 添加了 `def tokenize(self, text)`

- [x] The Initialization strategy seems a little different from the reference repo.  

+ Float16 问题导致，暂时忽略

- [x] change `embed_tokens` to optional args with default value None

- [x] We should save cache during decoder forwarding process rather than generating cache. 

- 将初始化 `cache` 的操作移动到了`blenderbotModel` 的前向传导中（Blenderbot 718行左右）。当且仅当在解码第一个时间步时，代码会根据 `encoder_output` 生成初始的 `cache`。对于其余时间步，`cache` 会自动更新与加载。

- [x] remove `pad_token_id` from `BlenderbotSmallLearnedPositionalEmbedding`.

- [x] We need to check whether decoder_input_ids is None （添加了：）

  ```python
  if decoder_input_ids is None:
  	raise ValueError("Decoder_input_ids cannot be None.")
  ```

- [x] Specify key for funcitons.

- [x] add a class named `BlenderbotForCausalLM`

+ **代码中添加了** `class BlenderbotDecoderLayer(nn.TransformerDecoderLayer)` 。（因为causalLM只使用了decoder部分，decoder layer中并没没有使用到 cross-attention。原 `nn.TransformerDecoderLayer` 不支持跳过 cross-attention ）
+ **代码中添加了** `class TransformerDecoder(nn.TransformerDecoder)` 。（`BlenderbotForCausalLM` 中并没有使用 `encoder_output`，所以传递的 `memory` 参数值为 `None`，使用 `paddle.nn.TransformerDecoder`  会出 bug。新写的这个类**仅**增加了对 memory = None 时候的判断。）
+ （Blenderbot 916-922行）由于  `BlenderbotForCausalLM` 中没有 cross-attention 操作，不需要使用到 `static_cache`，因此这边设置了使用一个全0张量来生成初始化 `cache`。
+ CausalLM前向传导核对：(文件与对比代码可以在 [github这里](https://github.com/kevinng77/blenderbot_paddle) 查看 )

[github这里](https://github.com/kevinng77/blenderbot_paddle) 根目录下的 `CLM_check.py` 为执行核对文件，运行代码核对：

```shell
python CLM_check.py --model_name=blenderbot_small-90M
python CLM_check.py --model_name=blenderbot-400M-distill
```

精度对齐时候，两个 `CausalLM` 模型都加载了 `ConditionalGeneration` 的预训练模型。由于架构与权重不完全匹配，会出现encoder的权重没被加载使用的警告。对齐结果如下：

```shell
huggingface facebook/blenderbot_small-90M vs paddle blenderbot_small-90M
input text: My friends are cool but they eat too many carbs.
mean difference: tensor(1.8442e-06)
max difference: tensor(1.2279e-05)

huggingface facebook/blenderbot-400M-distill vs paddle blenderbot-400M-distill
input text: My friends are cool but they eat too many carbs.
mean difference: tensor(1.9658e-06)
max difference: tensor(2.2888e-05)
```

## 其他修正

- [x] 参考 [hugging face代码](https://github.com/huggingface/transformers/blob/master/src/transformers/models/blenderbot/modeling_blenderbot.py) 的第955行， 修正了原 decoder 中 position embedding 的计算方式。 （参考 blenderbot/modeling 482-488）

+ 如果使用 cache 加速解码， decoder 生成 position embedding 时候会根据最新的 incremental_cache 的 key,values 长度（即已经生成的句子的长度）来调整 position embedding 的值。

```python
# cache[num_layer][0] is an instance of `MultiHeadAttention.Cache` containing
# k and v with shape of `[batch_size, num_heads, len_seq, embed_dim // num_heads]`
# ``len_seq`` refer to the length of ``decoder_input_ids``
# Refer to paddle.nn.MultiHeadAttention.gen_cache for more details regarding cache.
past_key_values_length = cache[0][0].k.shape[2] if cache is not None else 0

decoder_inputs_embed_pos = self.decoder_embed_positions(
            input_ids_shape=decoder_input_ids.shape,
            past_key_values_length=past_key_values_length)
```

- [ ] 此外 use_cache 相关的配置比较多。由于一般只在生成句子的时候才会使用 cache 进行加速，而paddle暂时不支持encoder-decoder模型的生成，所以我挺头疼不知道怎么进行 use_cache 这一方便的核对比较好。目前只是在调试模式下打个断点，核对了一下前后的数据流。
- [x] 用 `GPTTokenizer` 重新编写了 `blenderbotSmallTokenizer`。（一开始写`blenderbotSmallTokenizer` 时候参考了 hugging face 的方法，继承了 `PretrainedTokenizer` 。后来对比了一下发现 继承`GPTTokenizer` 代码可以少写很多行，阅读起来更清晰方便，修改后的 Tokenizer 也使用 之前的 [token_check](https://github.com/kevinng77/blenderbot_paddle/blob/master/tokenizer_check.py)  文件进行了核对。）

