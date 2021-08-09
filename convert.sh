for model in "blenderbot_small-90M" "blenderbot-400M-distill" "blenderbot-1B-distill" "blenderbot-3B"
do
  python convert.py --model_name=${model}
done
