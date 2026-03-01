python ./Run/pre_train.py \
    --config configs/ARROW/pretrain_one_step.yaml \
    --data.root_dir ./dataset \
    --trainer.logger.init_args.mode disabled

python ./Run/fine_tune_RL.py\
    --config configs/ARROW/finetune_RL.yaml \
    --model.root_dir ./dataset

python Run/inference_PL.py \
    --config configs/ARROW/inference.yaml \
    --data.root_dir ./dataset \
    --data.val_batch_size 4

python ./Run/inference_RL.py\
    --config configs/ARROW/inference_RL.yaml \
    --model.root_dir ./dataset