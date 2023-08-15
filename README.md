# kaggle hubmap 2023

## data download

```
$ kaggle datasets download -w kaerunantoka/hubmap-converted-to-coco-ds1-5fold

$ unzip hubmap-converted-to-coco-ds1-5fold.zip -d hubmap-converted-to-coco-ds1-5fold

$ rm hubmap-converted-to-coco-ds1-5fold.zip
```

## setup mmdet 3.0.0

```
$ pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

$ pip install mmcv==2.0.0rc4 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html

$ git clone https://github.com/open-mmlab/mmdetection.git

$ pip install "mmdet==3.0.0"

$ pip install "mmcls>=1.0.0rc6"
```

## create pseudo-labeled-coco-format-data

```
$ cd src
$ python create_pseudo_label_for_ds2_5fold.py -e 021 -th 0.8

$ python create_pseudo_label_for_ds3_5fold.py -e 062 -th 0.9 -dl True

$ python create_pseudo_label_for_ds2_5fold_late.py -e late001 -th 0.8
```

## training

- default run
```
$ python mmdetection/tools/train.py configs/hubmap/custom_config.py
```

- 5fold run (by original runner) and upload weights to kaggle dataset
```
$ cd src
$ python train_5fold.py configs/hubmap/exp038.py
```

- Multi GPU run
```
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=2 \
    --master_port=$PORT \
    train_5fold_ddp_late.py \
    configs/hubmap/late007.py \
    --launcher pytorch ${@:3}
```

## show mean mAP

```
# single case
$ python show_mean_map_from_mmdet_log.py -e exp074 -p work_dirs
$ python show_mean_map_from_mmdet_log.py -e exp078 -p /external_disk/work_dirs
$ python show_mean_map_from_mmdet_log.py -e late001 -p work_dirs

# multiple case
$ python show_mean_map_from_mmdet_log.py -e 'exp021 exp027' -p work_dirs
```

## upload output

```
$ kaggle datasets create --dir-mode zip -p /external_disk/work_dirs/exp021

$ kaggle datasets version --dir-mode zip -p /external_disk/work_dirs/exp065 -m "reduce pth"
```

## linter and formatter

```
$ make source-code-format
```