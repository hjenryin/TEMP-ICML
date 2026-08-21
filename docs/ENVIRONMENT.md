# Environment

We used Python 3.11, PyTorch 2.9 + CUDA 12.8.

## Create a matching env

```bash
conda create -y -n temp python=3.11
conda activate temp
pip install --upgrade pip

# PyTorch (CUDA 12.8 wheels)
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
  --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

If your machine has a different CUDA driver, install a compatible PyTorch build, but keep the rest of `requirements.txt` aligned when possible.

### LLaMA-Factory + DeepSpeed (SFT)

`llamafactory-cli` must be on `PATH` (provided by the `llamafactory` package). DeepSpeed ZeRO-2 config: `configs/ds_z2_config.json`.

FlashAttention 2 and `liger-kernel` are enabled in the SFT YAMLs.

### Math eval (evalchemy)

Clone [evalchemy](https://github.com/mlfoundations/evalchemy) and set:

```bash
export EVALCHEMY_DIR=/path/to/evalchemy
```


## Environment variables

```bash
export WANDB_API_KEY=...          # for SFT (LLaMA-Factory report_to: wandb)
export WANDB_PROJECT=temp
export TEMP_DATA_ROOT=$PWD/data
export TEMP_OUTPUT_ROOT=$PWD/outputs
export EVALCHEMY_DIR=...          # math only
export HF_HOME=...                # optional HF cache
```

Create a `.env` if you prefer; load it yourself before running scripts (scripts do not auto-load `.env` unless you use `python-dotenv` wrappers).
