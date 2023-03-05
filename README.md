# Guided-SR
#### Train:

> torchrun --nnodes 1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:10345 main.py -c ./config/nir.yml --scale 8 



#### Test

> torchrun --nnodes 1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:10345 main.py -c ./config/nir.yml --scale 8 --test_only
