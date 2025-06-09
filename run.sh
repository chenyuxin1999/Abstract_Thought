for lang in sw zh de fr th; do
    CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 \
    train.py \
    --language $lang \
    --base_model "Llama-3.1-8B" \
    --neuron_path "./neuron/llama-3.1-8B_shared_neuron_0.05.json" \
    --batch_size 128 \
    --micro_batch_size 4 \
    --num_epochs 1 \
    --learning_rate 5e-5 \
    --cutoff_len 1024 \
    --group_by_length False
done