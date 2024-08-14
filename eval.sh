CUDA_VISIBLE_DEVICES=6,7 python -m scripts.eval \
    --data_dir /data2/seongun/data/liver_tumor/datasets/L00_T20_V01 \
    --model_dir ./models/v2/transformed_v1/best_model.pt \
    --batch_size 128
