import glob
import pprint
from mmengine import Config
from mmdet.apis import init_detector, inference_detector

EXP_ID = '065'
work_dir_path = '/external_disk/work_dirs/exp065/fold0'
config_file = f'{work_dir_path}/exp{EXP_ID}.py'
cfg = Config.fromfile(config_file)
pprint.pprint(cfg.model)
print(cfg.model.test_cfg.rcnn.score_thr)
print(cfg.model.test_cfg.rcnn.max_per_img)
print(cfg.model.test_cfg.rcnn.nms.type)

cfg.model.test_cfg.rcnn.score_thr = 0.0001
cfg.model.test_cfg.rcnn.max_per_img = 300
cfg.model.test_cfg.rcnn.nms.type = 'soft_nms'
print(cfg.model.test_cfg.rcnn.score_thr)
print(cfg.model.test_cfg.rcnn.max_per_img)
print(cfg.model.test_cfg.rcnn.nms.type)


checkpoint_file = glob.glob(f'{work_dir_path}/best_coco_segm_mAP_epoch_*.pth')[-1]
# model = init_detector(config_file, checkpoint_file, device='cuda:0') 
model = init_detector(cfg, checkpoint_file, device='cuda:0') 

# print(model.cfg)