CUDA_VISIBLE_DEVICES=3,4 python -m scripts.eval \
    --data_dir ./datasets/L03_T05 \
    --model_dir ./models/resnet50/epoch_09.pt \
    --batch_size 128
