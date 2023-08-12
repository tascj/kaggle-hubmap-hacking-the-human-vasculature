# HuBMAP - Hacking the Human Vasculature

[Competition](https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature)
[Solution Write-up](https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature/discussion/429060)

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

## Prepare data

```
python tools/prepare_data.py
python tools/drop_dupliates.py
```

Follow [mmdet tutorial](https://mmdetection.readthedocs.io/en/latest/user_guides/dataset_prepare.html) to prepare coco dataset.

Directory structure should be as follows.

```
├── data
│   ├── coco
│   ├── dtrain0i.json
│   ├── dtrain1i.json
│   ├── dtrain_dataset2.json
│   ├── dtrain_dataset2_dropdup.json
│   ├── dtrainval.json
│   ├── dval0i.json
│   ├── dval1i.json
│   ├── polygons.jsonl
│   ├── sample_submission.csv
│   ├── test
│   ├── tile_meta.csv
│   ├── train
│   └── wsi_meta.csv
└── kaggle-hubmap-hacking-the-human-vasculature
    ├── LICENSE
    ├── README.md
    ├── configs
    ├── custom_modules
    ├── docker
    ├── test.py
    ├── tools
    ├── train.py
    └── work_dirs
```

## Training

### RTMDet-x (best single model)

```
python train.py configs/r0.py --amp
```

### Other models

```
python train.py configs/y0.py --amp

python train.py configs/m0.py --amp

python train.py configs/coco/sb.py --amp
python train.py configs/sb0.py --amp

python train.py configs/coco/s.py --amp
python train.py configs/s0.py --amp
```

## Inference

Check training log for best iteration and use `tools/dump_ckpt.py` to extract the best checkpoint.

Inference and ensemble could be found [here](https://www.kaggle.com/code/tascj0/hubmap-2023-release?scriptVersionId=139357383).
