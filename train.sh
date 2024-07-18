CUDA_VISIBLE_DEVICES=3,4 python -m scripts.train_full \
    --data_dir ./datasets/L03_T05 \
    --model_dir ./models/pretrained/resnet50_224_1.pth \
    --save_dir ./models/resnet50 \
    --batch_size 256 \
    --num_epochs 1000
