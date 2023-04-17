cd ./src
python main.py train --logger --augment --encoder se-resnet34 --loss adyolo --dataset DCASE2020 --device cuda:0
python main.py train --logger --augment --encoder se-resnet34 --loss adyolo --dataset DCASE2021 --device cuda:0
python main.py train --logger --augment --encoder se-resnet34 --loss adyolo --dataset DCASE2022 --device cuda:0
