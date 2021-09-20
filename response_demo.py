import paddle
import argparse


def main(model, tokenizer, strategy):
    model.eval()
    sample_text = "My friends are cool but they eat too many carbs."
    inputs = tokenizer(sample_text, return_attention_mask=True, return_token_type_ids=False)
    inputs = {k: paddle.to_tensor([v]) for (k, v) in inputs.items()}
    result_ids, scores = model.generate(input_ids=inputs['input_ids'],
                                        max_length=60,
                                        min_length=20,
                                        decode_strategy=strategy,
                                        num_beams=10,
                                        length_penalty=0.65)
    sequence_ids = result_ids.numpy().tolist()[0]
    bot_response = tokenizer.convert_ids_to_string(sequence_ids)
    print("User:\t", sample_text)
    print("bot:\t", bot_response)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='blenderbot_small-90M',
                        help="blenderbot_small-90M or blenderbot-400M-distill")
    parser.add_argument("--strategy", type=str, default='beam_search',
                        help="'beam_search','greedy_search' or 'sampling'")
    args = parser.parse_args()
    model_name = args.model_name
    strategy = args.strategy
    if model_name in ['blenderbot_small-90M']:
        from paddlenlp.transformers import BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration

        pretrained_model_name = "blenderbot_small-90M"
        tokenizer = BlenderbotSmallTokenizer.from_pretrained(pretrained_model_name)
        model = BlenderbotSmallForConditionalGeneration.from_pretrained(pretrained_model_name)

    elif model_name in ['blenderbot-400M-distill', 'blenderbot-1B-distill', 'blenderbot-3B']:
        from paddlenlp.transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

        pretrained_model_name = "blenderbot-400M-distill"
        tokenizer = BlenderbotTokenizer.from_pretrained(pretrained_model_name)
        model = BlenderbotForConditionalGeneration.from_pretrained(pretrained_model_name)
    else:
        raise f"model name not in " \
              f"{['blenderbot_small-90M', 'blenderbot-400M-distill','blenderbot-1B-distill', 'blenderbot-3B']} "

    main(model, tokenizer, strategy)
