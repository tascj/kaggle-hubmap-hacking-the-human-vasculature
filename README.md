# HuBMAP - Hacking the Human Vasculature

https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature

## Environment setup

Build docker image

```
bash .dev_scripts/build.sh
```

Set env variables

```
export DATA_DIR="/path/to/data"
export CODE_DIR="/path/to/this/repo"
```

Start a docker container
```
bash .dev_scripts/start.sh all
```


## Training

```
python tools/prepare_data.py
python tools/drop_dupliates.py
```

```
python train.py configs/r0.py --amp
```
