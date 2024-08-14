CUDA_VISIBLE_DEVICES=5,6 python -m scripts.train \
    --data_dir /data2/seongun/data/liver_tumor/datasets/L00_T20_W \
    --model_dir ./models/pretrained/resnet50_224_1.pth \
    --save_dir ./models/v4/Rot_Res \
    --batch_size 128 \
    --num_epochs 200