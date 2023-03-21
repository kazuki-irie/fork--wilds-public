# iWildCam and FMoW baselines (WILDS)

This repository was originally forked from the [official repository of WILDS datasets](https://github.com/p-lambda/wilds) (commit [7e103ed](https://github.com/p-lambda/wilds/commit/7e103ed051b54936ba11d246b564605c6019af56))

For general instructions, please refer to the original repositiory.

This repository contains code used to produce experimental results presented in:

[Improving Baselines in the Wild](https://openreview.net/forum?id=9vxOrkNTs1x)

Apart from minor edits, the only main changes we introduce are:
* `--validate_every` flag (default: `1000`) to specify the frequency (number of training steps) of cross-validation/checkpoint tracking.
* `sub_val_metric` option in the dataset (see `examples/configs/datasets.py`) to specify a secondary metric to be tracked during training. This activates additional cross-validation and checkpoint tracking for the specified metric.

## Results

NB: To reproduce the numbers from the paper, the right PyTorch version must be used.
All our experiments have been conducted using `1.9.0+cu102`, except for `+ higher lr` rows in Table 2/FMoW (which we ran for the camera-ready and for the public release) for which `1.10.0+cu102` was used.

The training scripts, logs, and model checkpoints for the best configurations from our experiments can be downloaded from [here for iWildCam & FMoW](https://drive.google.com/file/d/17Wwg9xJCsP1NLv5Ic5xO_FV8gVqIBChs/view?usp=sharing).

### iWildCam
CV based on "Valid F1"
| Split / Metric | mean (std) | 3 runs |
|---|---|---|
|IID Valid Acc | 82.5 (0.8)  |  [0.817, 0.835, 0.822]|
|IID Valid F1 |  46.7 (1.0)  |  [0.456, 0.481, 0.464]|
|IID Test Acc| 76.2 (0.1)   |  [0.762, 0.763, 0.761]|
|IID Test F1|  47.9 (2.1)   |  [0.505, 0.479, 0.453]|
|Valid Acc| 64.1 (1.7)      |  [0.644, 0.619, 0.661]|
|Valid F1|  38.3 (0.9)      |  [0.39, 0.371, 0.389]|
|Test Acc| 69.0 (0.3)       |  [0.69, 0.694, 0.687]|
|Test F1|  32.1 (1.2)       |  [0.338, 0.31, 0.314]|

CV based on "Valid Acc"
| Split / Metric | mean (std) | 3 runs |
|---|---|---|
|IID Valid Acc| 82.6 (0.7)  |  [0.836, 0.821, 0.822]
|IID Valid F1|  46.2 (0.9)  |  [0.472, 0.45, 0.464]
|IID Test Acc| 75.8 (0.4)   |  [0.76, 0.753, 0.761]
|IID Test F1|  44.9 (0.4)   |  [0.444, 0.45, 0.453]
|Valid Acc| 66.6 (0.4)      |  [0.666, 0.672, 0.661]
|Valid F1|  36.6 (2.1)      |  [0.369, 0.339, 0.389]
|Test Acc| 68.6 (0.3)       |  [0.688, 0.682, 0.687]
|Test F1|  28.7 (2.0)       |  [0.279, 0.268, 0.314]

### FMoW

CV based on "Valid Region"
| Split / Metric | mean (std) | 3 runs |
|---|---|---|
|IID Valid Acc|         63.9 (0.2)  |  [0.64, 0.636, 0.641]|
|IID Valid Region|  62.2 (0.5)  |  [0.623, 0.616, 0.628]|
|IID Valid Year|    49.8 (1.8)  |  [0.52, 0.475, 0.5]|
|IID Test Acc|     62.3 (0.2)   |  [0.626, 0.621, 0.621]|
|IID Test Region|  60.9 (0.6)   |  [0.617, 0.603, 0.606]|
|IID Test Year|    43.2 (1.1)   |  [0.438, 0.417, 0.442]|
|Valid Acc|     62.1 (0.0)      |  [0.62, 0.621, 0.621]|
|Valid Region|  52.5 (1.0)      |  [0.538, 0.513, 0.524]|
|Valid Year|    60.5 (0.2)      |  [0.602, 0.605, 0.608]|
|Test Acc|      55.6 (0.2)       |  [0.555, 0.554, 0.558]|
|Test Region|   34.8 (1.5)       |  [0.369, 0.334, 0.34]|
|Test Year|     50.2 (0.4)       |  [0.499, 0.498, 0.508]|


CV based on "Valid Acc"
| Split / Metric | mean (std) | 3 runs |
|---|---|---|
|IID Valid Acc|         64.0 (0.1)  |  [0.641, 0.639, 0.641]|
|IID Valid Region|  62.3 (0.4)  |  [0.623, 0.617, 0.628]|
|IID Valid Year|    50.8 (0.6)  |  [0.514, 0.509, 0.5]|
|IID Test Acc|     62.3 (0.4)   |  [0.628, 0.62, 0.621]|
|IID Test Region|  61.1 (0.6)   |  [0.62, 0.608, 0.606]|
|IID Test Year|    43.6 (1.4)   |  [0.45, 0.417, 0.442]|
|Valid Acc|     62.1 (0.0)      |  [0.621, 0.621, 0.621]|
|Valid Region|  51.4 (1.3)      |  [0.522, 0.496, 0.524]|
|Valid Year|    60.6 (0.3)      |  [0.608, 0.601, 0.608]|
|Test Acc|      55.6 (0.2)       |  [0.556, 0.554, 0.558]|
|Test Region|   34.2 (1.2)       |  [0.357, 0.329, 0.34]|
|Test Year|     50.2 (0.5)       |  [0.496, 0.501, 0.508]|



## BibTex
```
@inproceedings{irie2021improving,
      title={Improving Baselines in the Wild}, 
      author={Kazuki Irie and Imanol Schlag and R\'obert Csord\'as and J\"urgen Schmidhuber},
      booktitle={Workshop on Distribution Shifts, NeurIPS},
      address={Virtual only},
      year={2021}
}
```
