# AD-YOLO: You Only Look Once in Training Multiple Sound Event Localization and Detection (ICASSP 2023)
This is a Pytorch implementation of [AD-YOLO: You Only Look Once in Training Multiple Sound Event Localization and Detection](https://doi.org/10.48550/arXiv.2303.15703).
We share an overall framework used to train and evaluate models/formats on DCASE 2020~2022 Task3 (SELD) datasets.


AD-YOLO tackles the SELD problem under an unknown polyphony environment.
Taking the notion of angular distance, we adapt the approach of [You Only Look Once](https://doi.org/10.48550/arXiv.1506.02640) (YOLO) algorithm to SELD.
Experimental results demonstrate the potential of AD-YOLO to outperform the existing formats and show the robustness of handling class-homogenous polyphony.


Below figure depicts an example how AD-YOLO designates the responsible predictions for each ground truth targets at a single time frame.
<p align="center">
<img src="/img/ADYOLO_responsibles.png" width="550" height="550">
</p>

## Environment & Python Requirements

* CUDA >= 11.0
* Python==3.8.11
* [Pytorch==1.10.0](https://pytorch.org/get-started/previous-versions/#v1100), with torchvision==0.11.0 & torchaudio==0.10.0

Use the [requirements.txt](/requirements.txt) to install the rest of Python dependencies.

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


[AD-YOLO/data/DCASE2020_SELD](/data/DCASE2020_SELD)
; [DCASE2021_SELD](/data/DCASE2021_SELD)
; [DCASE2022_SELD](/data/DCASE2022_SELD)

### 2. Preprocess Train Data
The first Python command below will slice the audio/label of training data into uniform time chunks.
You can give a specific annual dataset as an argument, such as "DCASE2020", "DCASE2021" and "DCASE2022".


If you give "scaler" as an action, this will compute and save the stats, mean and standard deviation, of acoustic feature from training data.

Hyperparameters stated in data configurations (e.g. [hyp_data_DCASE2022.yaml](/src/configs/hyp_data_DCASE2022.yaml)) involves with this procedure.

```bash
python src/preprocess.py chunking --dataset all
python src/preprocess.py scaler --dataset all
```

### 3-1. Initiate the Model Training Pipeline

If you want to initiate the pipeline directly, use as an example below:
```bash
cd ./src
python main.py train --encoder se-resnet34 --loss adyolo --dataset DCASE2021 --device cuda:0
```

Or you would manage the experiment easier using [run.sh](/run.sh).
```bash
sh run.sh
```

You can find more detailed description for command arguments in [src/main.py](/src/main.py) (see also [src/configs/](/src/configs) for hyperparameters).
```bash
python main.py -h
```

The pipeline will first create the folder to save the setups, result, model weights and checkpoint of the experiment. 

You can check them from [src/results/](/src/results). 

### 3-2. Resume the Interrupted Training

You can restart the pipeline from the checkpoint with the name of the experiment folder.


### 3-3. Evalutate the Experimental Result.

You can also use the name of the result folder to evaluate the best-validated model.

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

