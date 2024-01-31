# Guided-SR
#### Train:

torchrun --nnodes 1 --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:10342 main.py -c ./config/nir.yml --scale 8 --model_name Base2 --show_every 10 --epochs 90 --num_gpus 4 --opt Adam --decay_epochs '40_70' --lr 3e-4 --embed_dim 64 --sched multistep --seed 60



#### Test

> torchrun --nnodes 1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:10345 main.py -c ./config/nir.yml --scale 8 --test_only
