import argparse
import paddle

paddle.set_device("cpu")

text = [
    # "it is a nice day today!",  # 官方例题
    "My friends are cool but they eat too many carbs.",  # 官方例题
    "My 'but' they:@ eat too many carbs:)",
]


def run_check(model_name):
    for t in text:
        print("input text:", t)
        PTtokenizer = PTTokenizer.from_pretrained('facebook/' + model_name)
        pt_temp = PTtokenizer(t)
        pt_inputs = pt_temp["input_ids"]
        print("torch tokenizer: ", pt_inputs)

        PDtokenizer = PDTokenizer.from_pretrained(model_name)
        pd_temp = PDtokenizer(t)
        pd_inputs = pd_temp["input_ids"]
        print("paddle tokenizer: ", pd_inputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='blenderbot_small-90M',
                        help="blenderbot_small-90M or blenderbot-400M-distill")
    args = parser.parse_args()
    model_name = args.model_name
    if model_name == 'blenderbot_small-90M':
        from paddlenlp.transformers import BlenderbotSmallTokenizer as PDTokenizer
        from transformers import BlenderbotSmallTokenizer as PTTokenizer

    elif model_name == 'blenderbot-400M-distill':
        from paddlenlp.transformers import BlenderbotTokenizer as PDTokenizer
        from transformers import BlenderbotTokenizer as PTTokenizer
    else:
        raise f"model name not in {['blenderbot_small-90M', 'blenderbot-400M-distill']} "

    run_check(model_name=model_name)
