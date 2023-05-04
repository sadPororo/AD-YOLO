# AD-YOLO: You Only Look Once in Training Multiple Sound Event Localization and Detection (ICASSP 2023)
This is a Pytorch implementation of [AD-YOLO: You Only Look Once in Training Multiple Sound Event Localization and Detection](https://doi.org/10.48550/arXiv.2303.15703).
We share an overall framework used to train and evaluate models/formats on DCASE 2020~2022 Task3 (SELD) datasets.


AD-YOLO tackles the SELD problem under an unknown polyphony environment.
Taking the notion of angular distance, we adapt the approach of [You Only Look Once](https://doi.org/10.48550/arXiv.1506.02640) (YOLO) algorithm to SELD.
Experimental results demonstrate the potential of AD-YOLO to outperform the existing formats and show the robustness of handling class-homogenous polyphony.

## Environment

Python==3.8.11

CUDA >= 11.0

Soundfile==0.10.3.post1


## Requirements

Use the "requirements.txt" file

```bash
pip install -r requirements.txt
```

## Usage

### Prepare Datasets

### Preprocess Train Data
The datasets can be downloaded from here:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4064792.svg)](https://doi.org/10.5281/zenodo.4064792) TAU-NIGENS Spatial Sound Events 2020
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5476980.svg)](https://doi.org/10.5281/zenodo.5476980) TAU-NIGENS Spatial Sound Events 2021
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6406873.svg)](https://doi.org/10.5281/zenodo.6406873) [DCASE2022 Task 3] Synthetic SELD mixtures for baseline training
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6600531.svg)](https://doi.org/10.5281/zenodo.6600531) STARSS22: Sony-TAu Realistic Spatial Soundscapes 2022 dataset



For the detailed file hierarchies and structures, please refer to:

### Train/Evaluate

You can refer to "run.sh" file and look for the arguments in "./src/main.py" file

```bash
sh run.sh
```

## Citation
```
@article{kim2023adyolo,
  title={AD-YOLO: You Only Look Once in Training Multiple Sound Event Localization and Detection},
  author={Kim, Jin Sob and Park, Hyun Joon and Shin, Wooseok and Han, Sung Won},
  journal={arXiv preprint arXiv.2303.15703},
  year={2023}
}
```
To appear in [ICASSP2023](https://2023.ieeeicassp.org/)


## [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) License
This repository is released under the [MIT](https://choosealicense.com/licenses/mit/) license.

The file ```src/utils/seld_metrics.py``` was adapted from the [sharathadavanne/seld-dcase2022](https://github.com/sharathadavanne/seld-dcase2022), released under the MIT license. We modified some parts to fit the repository structure and added some classes & functions for exclusive evaluation under polyphony circumstances.

