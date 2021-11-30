# iWildCam and FMoW baselines (WILDS)

This repository was originally forked from the [official repository of WILDS datasets](https://github.com/p-lambda/wilds) (commit [7e103ed](https://github.com/p-lambda/wilds/commit/7e103ed051b54936ba11d246b564605c6019af56))

For general instructions, please refer to the original repositiory.

This repository contains code used to produce experimental results presented in:

[Improving Baselines in the Wild](openreview) (link coming soon)

Apart from minor edits, the only main changes we introduce are:
* `--validate_every` flag (default: `1000`) to specify the frequency (number of training steps) of cross-validation/checkpoint tracking.
* `sub_val_metric` option in the dataset (see `examples/configs/datasets.py`) to specify a secondary metric to be tracked during training. This activates additional cross-validation and checkpoint tracking for the specified metric.

NB: To reproduce the numbers from the paper, the right PyTorch version must be used.
All our experiments have been conducted using `1.9.0+cu102`, except for `+ higher lr` rows in Table 2/FMoW (which we ran for the camera-ready and for the public release) for which `1.10.0+cu102` was used.

The training scripts, logs, and model checkpoints for the best configurations from our experiments can be found here for [iWildCam & FMoW](https://people.idsia.ch/~kazuki/work/wilds).


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
