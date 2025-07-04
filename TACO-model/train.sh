CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    torchrun --nproc_per_node=8 main.py \
    --base=configs/train_vac.yaml \
    --name=Video_Amodal --seed=1234 --num_nodes=1 --wandb=1 \
    model.base_learning_rate=2e-5 \
    model.params.optimizer_config.params.foreach=False \
    data.params.dset_root=/path/to/OvO_dataset \
    data.params.frame_width=384 \
    data.params.frame_height=384 \
    data.params.batch_size=4 \
    data.params.num_workers=8 \
    data.params.data_gpu=0 \
    data.params.train_path=/path/to/training_json \
    data.params.val_path=/path/to/validation_json \
    lightning.callbacks.image_logger.params.batch_frequency=200 \
    lightning.trainer.devices="0,1,2,3,4,5,6,7" 