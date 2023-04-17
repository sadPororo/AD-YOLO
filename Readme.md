This is an overall framework to train and evaluate models/formats on DCASE 2020~2022 Task3 (SELD) datasets.


# Environment

Python==3.8.11

CUDA >= 11.0

Soundfile==0.10.3.post1


## Requirements

Use the "requirements.txt" file

```bash
pip install -r requirements.txt
```

## Usage

You can refer to "run.sh" file and look for the arguments in "./src/main.py" file

```bash
sh run.sh
```

## Citation
```
To appear in ICASSP2023
```
https://2023.ieeeicassp.org/


## License
This repository is released under the [MIT](https://choosealicense.com/licenses/mit/) license.

The file ```src/utils/seld_metrics.py``` was adapted from the [sharathadavanne/seld-dcase2022](https://github.com/sharathadavanne/seld-dcase2022), released under the MIT license. We modified some parts to fit the repository structure and added some classes & functions for exclusive evaluation under polyphony circumstances.

