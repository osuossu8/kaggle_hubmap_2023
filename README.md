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

## training

- default run
```
$ python mmdetection/tools/train.py configs/hubmap/custom_config.py
```

- 5fold run (by original runner) and upload weights to kaggle dataset
```
$ cd src
$ python train_5fold.py configs/hubmap/exp012.py
```

## upload output

```
$ kaggle datasets create --dir-mode zip -p work_dirs/exp012
```