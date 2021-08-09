import argparse
import paddle
import torch

paddle.set_device("cpu")
text = [
    # "it is a nice day today!",  # 官方例题
    "My friends are cool but they eat too many carbs.",  # 官方例题
]


def run_check(model_name):
    PDtokenizer = PDTokenizer.from_pretrained(model_name)

    for t in text:
        temp = PDtokenizer(t)
        inputs = temp["input_ids"]  # tokenizer consistency could be check in tokenizer_check file
        pt_inputs = torch.tensor([inputs])
        pd_inputs = paddle.to_tensor([inputs])

        # torch model
        pt_model = PTmodel.from_pretrained('facebook/' + model_name)
        shifted_input_ids = torch.zeros_like(pt_inputs)
        shifted_input_ids[:, 1:] = pt_inputs[:, :-1].clone()
        shifted_input_ids[:, 0] = 1
        pt_model.eval()
        with torch.no_grad():
            pt_outputs = pt_model(input_ids=pt_inputs,
                                  decoder_input_ids=shifted_input_ids).logits

        # paddle model
        pd_model = PDmodel.from_pretrained(model_name)
        pd_model.eval()
        with paddle.no_grad():
            outputs = pd_model(pd_inputs).numpy()
            pd_outputs = torch.from_numpy(outputs)
        print(f"huggingface {'facebook/' + model_name} vs paddle {model_name}")
        print(f"input text: {t}")
        print("mean difference:", torch.mean(torch.abs(pt_outputs - pd_outputs)))
        print("max difference:", torch.mean(torch.max(pt_outputs - pd_outputs)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='blenderbot-400M-distill',
                        help="blenderbot_small-90M or blenderbot-400M-distill")
    args = parser.parse_args()
    model_name = args.model_name
    if model_name in ['blenderbot_small-90M','blenderbot-90M']:
        from paddlenlp.transformers import \
            BlenderbotSmallTokenizer as PDTokenizer, \
            BlenderbotSmallForConditionalGeneration as PDmodel
        from transformers import \
            BlenderbotSmallForConditionalGeneration as PTmodel

    elif model_name in ['blenderbot-400M-distill','blenderbot-1B-distill']:
        from paddlenlp.transformers import \
            BlenderbotTokenizer as PDTokenizer, \
            BlenderbotForConditionalGeneration as PDmodel

        from transformers import \
            BlenderbotTokenizer as PTTokenizer, \
            BlenderbotForConditionalGeneration as PTmodel
    else:
        raise f"model name not in {['blenderbot_small-90M', 'blenderbot-400M-distill']} "

    run_check(model_name=model_name)
