# Guided-SR
#### Train:

> torchrun --nnodes 1 --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:10342 main.py -c ./config/nir.yml --scale 4 --model_name Base2 --show_every 10 --epochs 120 --num_gpus 4 --opt Adam --decay_epochs '70_90' --lr 3e-4 --embed_dim 64 --sched multistep --seed 60; torchrun --nnodes 1 --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:10342 main.py -c ./config/nir.yml --scale 8 --model_name Base2 --show_every 10 --epochs 81 --num_gpus 4 --opt Adam --decay_epochs '40_70' --lr 3e-4 --embed_dim 64 --sched multistep --seed 60 --pre_train --load_name model_000040.pth



#### Test

> torchrun --nnodes 1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:10345 main.py -c ./config/nir.yml --scale 8 --test_only
