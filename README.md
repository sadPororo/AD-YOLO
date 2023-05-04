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

### 1. Prepare Datasets

The datasets can be downloaded from here:


* TAU-NIGENS Spatial Sound Events 2020 [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4064792.svg)](https://doi.org/10.5281/zenodo.4064792)


* TAU-NIGENS Spatial Sound Events 2021 [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5476980.svg)](https://doi.org/10.5281/zenodo.5476980)


* [DCASE2022 Task 3] Synthetic SELD mixtures for baseline training [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6406873.svg)](https://doi.org/10.5281/zenodo.6406873)


* STARSS22: Sony-TAu Realistic Spatial Soundscapes 2022 dataset [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6600531.svg)](https://doi.org/10.5281/zenodo.6600531)



For detailed information on file hierarchies and structures, please see:


[AD-YOLO/data/DCASE2020_SELD](https://github.com/sadPororo/AD-YOLO/tree/main/data/DCASE2020_SELD)
; [DCASE2021_SELD](https://github.com/sadPororo/AD-YOLO/tree/main/data/DCASE2021_SELD)
; [DCASE2022_SELD](https://github.com/sadPororo/AD-YOLO/tree/main/data/DCASE2022_SELD)

### 2. Preprocess Train Data
The Python command below will slice the audio/label of training data into uniform time chunks.
You can give a specific annual dataset as an argument, such as "DCASE2020", "DCASE2021" and "DCASE2022".


If you give "scaler" as an action, this will compute and save the stats, mean and standard deviation, of acoustic feature from training data.


```bash
python src/preprocess.py chunking --dataset all
python src/preprocess.py scaler --dataset all
```

### 3. Train/Evaluate Model

If you want to initiate the pipeline directly, use as an example below:
```bash
cd ./src
python main.py train --encoder se-resnet34 --loss adyolo -- dataset DCASE2021 --device cuda:0
```

Or you would manage the experiment easier using [run.sh](https://github.com/sadPororo/AD-YOLO/blob/main/run.sh).
```bash
sh run.sh
```

You can find more detailed description for command arguments in [src/main.py](https://github.com/sadPororo/AD-YOLO/blob/main/src/main.py).
```bash
python main.py -h
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


## License [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
This repository is released under the [MIT](https://choosealicense.com/licenses/mit/) license.

The file ```src/utils/seld_metrics.py``` was adapted from the [sharathadavanne/seld-dcase2022](https://github.com/sharathadavanne/seld-dcase2022), released under the MIT license. We modified some parts to fit the repository structure and added some classes & functions for exclusive evaluation under polyphony circumstances.

