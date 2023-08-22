# PyTorch Implementation of Person Search
This includes OIMNet, OIMNet++, NAE4PS

## Requirements
* Python 3.8
* PyTorch 1.7.1
* GPU memory >= 22GB

## Getting Started
First, clone our git repository.

### Prepare datasets
Download [PRW](https://github.com/liangzheng06/PRW-baseline) and [CUHK-SYSU](https://github.com/ShuangLI59/person_search) datasets.<br>
Modify the dataset directories below if necessary.

* PRW: L4 of [configs/prw.yaml](https://github.com/cvlab-yonsei/OIMNetPlus/blob/main/configs/prw.yaml)<br>
* CUHK-SYSU: L3 of [configs/ssm.yaml](https://github.com/cvlab-yonsei/OIMNetPlus/blob/main/configs/ssm.yaml)<br>

Your directories should look like:
```
    <working_dir>
    OIMNetPlus
    ├── configs/
    ├── datasets/
    ├── engines/
    ├── losses/
    ├── models/
    ├── utils/
    ├── defaults.py
    ├── Dockerfile
    └── train.py
    
    <dataset_dir>
    ├── CUHK-SYSU/
    │   ├── annotation/
    │   ├── Image/
    │   └── ...
    └── PRW-v16.04.20/
        ├── annotations/
        ├── frames/
        ├── query_box/
        └── ...
```

## Training and Evaluation

By running the commands below, evaluation results and training losses will be logged into a .txt file in the output directory.<br>
Or open run.ipynb for commands.

* OIMNet++<br> 
    `$ python train.py --cfg configs/prw.yaml`<br>
    `$ python train.py --cfg configs/ssm.yaml` 

* OIMNet+++<br>
    `$ python train.py --cfg configs/prw.yaml MODEL.ROI_HEAD.AUGMENT True`<br>
    `$ python train.py --cfg configs/ssm.yaml MODEL.ROI_HEAD.AUGMENT True`

* OIMNet<br>
    `$ python train.py --cfg configs/prw.yaml MODEL.ROI_HEAD.NORM_TYPE 'none' MODEL.LOSS.TYPE 'OIM'`<br> 
    `$ python train.py --cfg configs/ssm.yaml MODEL.ROI_HEAD.NORM_TYPE 'none' MODEL.LOSS.TYPE 'OIM'`
  
* NAE4PS<br>
    `$ python train.py --cfg configs/prw.yaml MODEL.ROI_HEAD.NORM_TYPE 'none' MODEL.LOSS.TYPE 'OIM' MODEL.ROI_HEAD.EMBD 'NAE'`<br> 
    `$ python train.py --cfg configs/ssm.yaml MODEL.ROI_HEAD.NORM_TYPE 'none' MODEL.LOSS.TYPE 'OIM' MODEL.ROI_HEAD.EMBD 'NAE'`

> We support training/evaluation using **single** GPU only. <br>
> This is due to unsynchronized items across multiple GPUs in OIM loss (i.e., LUT and CQ) and ProtoNorm. <br>

## Credits
Our person search implementation is heavily based on [Di Chen](https://di-chen.me/)'s [NAE](https://github.com/dichen-cd/NAE4PS) and [Zhengjia Li](https://github.com/serend1p1ty)'s [SeqNet](https://github.com/serend1p1ty/SeqNet) and [Sanghoon Lee](https://github.com/sanghoooon)'s [OIMNet++](https://github.com/sanghoooon/OIMNetPlus).<br>
ProtoNorm implementation is based on [ptrblck](https://github.com/ptrblck)'s manual BatchNorm implementation [here](https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py).
