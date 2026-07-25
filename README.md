# Fast-CenLaneNet

**Fast-CenLaneNet: A Lightweight Instance Segmentation-Based Network for Real-Time Lane Detection**

The method is built on the CenLaneNet framework and keeps the center-based instance segmentation pipeline while making the network lighter for real-time lane detection.

## Overview

Fast-CenLaneNet follows an instance segmentation-based lane detection paradigm.
The implementation in this repository includes:

- a lightweight backbone based on Ghost convolutions and Self-BN,
- learnable spatial similarity attention for lane-related feature enhancement,
- Ghost convolution-based multi-branch prediction heads.

## Dataset Preparation

Download TuSimple from the official benchmark:

```text
https://github.com/TuSimple/tusimple-benchmark
```

Expected dataset layout:

```text
/path/to/tusimple/
|-- train_set/
|   |-- clips/
|   `-- ...
`-- test_set/
    |-- clips/
    |-- test_tasks_0627.json
    `-- ...
```

## Training

Train from scratch on TuSimple:

```bash
python Train.py --GtDataroot /path/to/tusimple/train_set/
```

## Testing on TuSimple

Download the pretrained model:

```text
Link: https://pan.baidu.com/s/12Re_kIcS4Paq4JFFIODyHg
Code: tcdu
```
Evaluating tusimple:

```bash
python Test_tusimple.py \
  --TusimpleTesting_root /path/to/tusimple/test_set/ \
  --testmodel model.pkl
```

## Citation

```bibtex
@article{han2026fastcenlanenet,
  title={Fast-CenLaneNet: A Lightweight Instance Segmentation-Based Network for Real-Time Lane Detection},
  author={Han, Qidong and Feng, Shuo and Gao, Yang and Li, Mengyao and Meng, Teng and Li, Ke and Yang, Yuhao},
  journal={Journal of Imaging},
  volume={12},
  number={7},
  pages={320},
  year={2026},
  doi={10.3390/jimaging12070320}
}
```

## Acknowledgements

This repository is developed on top of the ideas and code structure of **CenLaneNet**. We sincerely thank the CenLaneNet authors for their excellent work on center-based lane instance segmentation, which provides the foundation for this project.

CenLaneNet: https://github.com/SYVAE/CenLaneNet

We also thank the TuSimple benchmark and the open-source lane detection community for providing datasets, evaluation protocols, and reference implementations.
