# in-memory_reasoning

### Installation
```
pip install -r requirements.txt
```
You may want to create two folders named model and dataset, putting your models under the model folder or gitting from HuggingFace, changing the path with config.py. Dataset can be put under dataset folder, we're using LongMemEval which you can download from https://github.com/xiaowu0162/LongMemEval/tree/main.
### Train
```
torh run python train.py
```
### Inference
```
python inference.py "Your query here" --actor_ckpt <actor_checkpoint_name.pth> --reward_ckpt <reward_checkpoint_name.pth> [--ind_mem "Initial individual memory"] [--shared_mem "Initial shared memory"]
```
### Evaluation
```
python evaluation.py \
    --actor_ckpt actor_epoch_3.pth \
    --reward_ckpt reward_epoch_3.pth \
    --eval_split validation \
    --max_samples 200 \
    --threshold 0.65
```
