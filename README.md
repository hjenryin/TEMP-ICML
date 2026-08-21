# TEMP: Token-Efficient Model Perturbation for Reasoning Data Selection

Official code for **[Reasoning Quality Emerges Early: Data Curation for Reasoning Models](https://arxiv.org/abs/2606.26797)** (ICML 2026).

Project page: https://bigml-cs-ucla.github.io/TEMP-project-page/


## Repository layout

```
.
├── configs/                  # Selection defaults + LLaMA-Factory SFT YAMLs
├── data/                     # Place training / eval datasets here (see docs/DATASETS.md)
├── docs/                     # Environment, datasets, reproduction
└── TEMP/                     # Main code folder
    ├── select-high-loss.sh   # Step: difficulty filter
    ├── select-custom-path.sh # Step: diversity + brittle select
    ├── select-baseline.sh    # Baselines (perplexity / embedding / learnability)
    ├── run-m23k.sh / run-openthoughts-math.sh
    ├── eval-23k-llama-factory.sh
    └── src/
        ├── selection/        # selection.py, selector.py, sample_random.py
        ├── tokenize_data/    # m23k_to_sharegpt.py, openthoughts_to_sharegpt.py
        ├── train/            # llama_factory.py + SFT helpers / accuracy extractors
        ├── perturb/          # Random + directional perturbation
        ├── process_loss/     # Tokenwise loss + embedding baselines
        ├── eval/             # Medical benchmark eval (sglang)
```


## Setup

See [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) for conda / CUDA / sglang / LLaMA-Factory / FAISS.

```bash
git clone <this-repo> && cd <repo>
pip install -r requirements.txt
# plus LLaMA-Factory, FAISS, sglang as documented
export WANDB_API_KEY=...   # for SFT logging (LLaMA-Factory)
```

Place datasets under `data/` (or set `TEMP_DATA_ROOT`). See [docs/DATASETS.md](docs/DATASETS.md).


## Quick start

See [docs/REPRODUCTION.md](docs/REPRODUCTION.md).



## Citation

```bibtex
@inproceedings{jin2026reasoning,
  title={Reasoning Quality Emerges Early: Data Curation for Reasoning Models},
  author={Jin, Hongyi Henry and Yang, Wenhan and Ghaffari, Meysam and Morato, Carlos and Mirzasoleiman, Baharan},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026},
  url={https://openreview.net/forum?id=4Mu4AA14jr}
}
```

