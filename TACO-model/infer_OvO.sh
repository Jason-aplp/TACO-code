CUDA_VISIBLE_DEVICES=0 python scripts/infer_OvO.py --gpus=0 \
    --config_path=configs/infer_vac_OvO.yaml \
    --model_path=/path/to/your/model.ckpt \
    --output=../output \
    --frame_start=0 --frame_stride=2 --frame_rate=12 \
    --autocast=1 --num_samples=1 --num_steps=50