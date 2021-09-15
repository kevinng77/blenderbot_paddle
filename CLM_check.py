import argparse
import paddle
import torch

paddle.set_device("cpu")
text = [
    # "My friends are cool but they eat too many carbs.",  # 官方例题
    "nice day today!",  # 官方例题
    # "Hello, my dog is cute"
]


def run_check(model_name):
    PDtokenizer = PDTokenizer.from_pretrained(model_name)
    PTtokenizer = PTTokenizer.from_pretrained('facebook/' + model_name)
    for t in text:
        inputs =  PDtokenizer(t,return_attention_mask=True, return_token_type_ids=False)# tokenizer consistency could be check in tokenizer_check file
        pt_inputs = PTtokenizer(t, return_tensors="pt")
        pd_inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
        # torch model
        def get_torch_result(model_name):
            pt_model = PTmodel.from_pretrained('facebook/' + model_name)
            # shifted_input_ids = torch.zeros_like(pt_inputs["input_ids"])
            # shifted_input_ids[:, 1:] = pt_inputs["input_ids"][:, :-1].clone()
            # shifted_input_ids[:, 0] = 1
            pt_model.eval()
            with torch.no_grad():
                pt_outputs = pt_model(**pt_inputs).logits
            return pt_outputs

        def get_paddle_result(model_name):
            # paddle model
            pd_model = PDmodel.from_pretrained(model_name)
            pd_model.eval()
            with paddle.no_grad():
                # outputs = pd_model(pd_inputs)
                outputs = pd_model(**pd_inputs, use_cache=True)[0]
                pd_outputs = torch.from_numpy(outputs.numpy())
            return pd_outputs

        pt_outputs = get_torch_result(model_name)
        pd_outputs = get_paddle_result(model_name)
        print(f"huggingface {'facebook/' + model_name} vs paddle {model_name}")
        print(f"input text: {t}")
        print("mean difference:", torch.mean(torch.abs(pt_outputs - pd_outputs)))
        print("max difference:", torch.max(torch.abs(pt_outputs - pd_outputs)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='blenderbot_small-90M',
                        help="blenderbot_small-90M or blenderbot-400M-distill")
    args = parser.parse_args()
    model_name = args.model_name
    if model_name in ['blenderbot_small-90M', 'blenderbot-90M']:
        from paddlenlp.transformers import \
            BlenderbotSmallTokenizer as PDTokenizer, \
            BlenderbotSmallForCausalLM as PDmodel
        from transformers import \
            BlenderbotSmallTokenizer as PTTokenizer, \
            BlenderbotSmallForCausalLM as PTmodel

    elif model_name in ['blenderbot-400M-distill', 'blenderbot-1B-distill', 'blenderbot-3B']:
        from paddlenlp.transformers import \
            BlenderbotTokenizer as PDTokenizer, \
            BlenderbotForCausalLM as PDmodel

        from transformers import \
            BlenderbotTokenizer as PTTokenizer, \
            BlenderbotForCausalLM as PTmodel
    else:
        raise f"model name not in {['blenderbot_small-90M', 'blenderbot-400M-distill']} "

    run_check(model_name=model_name)
