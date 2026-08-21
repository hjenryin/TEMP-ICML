import numpy as np
import random
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OUT = os.environ.get("TEMP_OUTPUT_ROOT", str(_REPO_ROOT / "outputs"))

# OpenThoughts-Math correct subset size (see docs/DATASETS.md)
_N_EXAMPLES = 56370

# Create directories
os.makedirs(os.path.join(_OUT, "selection-openthoughts-math/random2k-1/data/"), exist_ok=True)
os.makedirs(os.path.join(_OUT, "selection-openthoughts-math/random2k-2/data/"), exist_ok=True)

# First sample with seed 42
random.seed(42)
sample1 = random.sample(range(_N_EXAMPLES), 2000)
np.save(os.path.join(_OUT, "selection-openthoughts-math/random2k-1/data/labeled_idx.npy"), np.array(sample1))

# Second sample with seed 123
random.seed(123)
sample2 = random.sample(range(_N_EXAMPLES), 2000)
np.save(os.path.join(_OUT, "selection-openthoughts-math/random2k-2/data/labeled_idx.npy"), np.array(sample2))

print("Samples saved successfully.")
