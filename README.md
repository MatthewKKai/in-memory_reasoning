# in-memory_reasoning

### Installation
```
pip install -r requirements.txt
```
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
