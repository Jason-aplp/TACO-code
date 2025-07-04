CUDA_VISIBLE_DEVICES=0 python scripts/infer.py --gpus=0 \
    --config_path=configs/infer_vac.yaml \
    --model_path=/path/to/your/model.ckpt \
    --output=../output \
    --frame_start=0 --frame_stride=2 --frame_rate=12 \
    --input_example=wild_minion \
    --autocast=1 --num_samples=10 --num_steps=50
