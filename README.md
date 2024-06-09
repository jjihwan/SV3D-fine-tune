# SV3D fine-tuning
Fine-tuning code for SV3D

## Setting up

#### PyTorch 2.0

```shell
conda activate sv3d python==3.10.14
pip3 install -r requirements.txt
```

#### Install `deepspeed` for training
```shell
pip3 install deepspeed
```


## Get checkpoints
Store them as following structure:
```
cd Novel-View-Refinement
    .
    └── checkpoints
        └── sv3d_p.safetensors
        └── sv3d_u.safetensors # might not be used.
```


## Training (WIP)
```shell
sh scripts/sv3d_finetune.sh
```


## Inference
Store the input images in `assets`
```shell
sh scripts/inference.sh
```
