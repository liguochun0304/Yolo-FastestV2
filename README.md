# ***FastestDet: it has higher accuracy and faster speed than Yolo-fastest https://github.com/dog-qiuqiu/FastestDet***



# 目标检测与分类实战
根据github上的一个[Yolo-FastestV2](https://github.com/dog-qiuqiu/Yolo-FastestV2)参考作为参考，以及[gd32ai-modelzoo](https://github.com/HomiKetalys/gd32ai-modelzoo/tree/main/object_detection/yolo_fastestv2)的代码进行复现，选用[TT100k交通标志检测数据集](https://cg.cs.tsinghua.edu.cn/traffic-sign/)完成检测与分类的任务。模型需要部署到开发板上，对推理速度有一定的要求，最后选用YoloFastestV2模型。

## 数据的预处理
因为我们需要将模型部署到gd32开发板上，而摄像头采用ov7670采集图像最大为640\*480,而数据集的图片大小为2048\*2048，所以我们将数据集的图像需要重新做切割。我按照每个图片的每一个标签的中心点做切割，切割成320\*240大小的图像，这样将多目标的数据集转换成了单目标检测的数据集。再将标签数据归一化，转换成了yolo所需的txt文件标签。

![文件目录](assets/1717839333124.png)

```python
# -*- coding: UTF-8 -*-

# @Project ：group_first 
# @File    ：preprocessing.py
# @IDE     ：PyCharm 
# @Author  ：liguochun0304@163.com
# @Date    ：2024/5/7 15:55

import json
import random

from PIL import ImageDraw, Image
import os
data_path = './data/annotations.json'
train_path = './data/train'
test_path = './data/test'

def save_json(save_path,data):
    assert save_path.split('.')[-1] == 'json'
    with open(save_path,'w') as file:
        json.dump(data,file)

def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)
def print_progress_bar(current, total, prefix='', suffix='', decimals=1, length=50, fill='█', print_end="\r"):
    """
    调用时需要提供当前进度和总进度。
    @params:
        current     - 当前的进度值（Int）
        total       - 总进度值（Int）
        prefix      - 前缀字符串（Str）
        suffix      - 后缀字符串（Str）
        decimals    - 百分比精度（Int）
        length      - 进度条长度（Int）
        fill        - 填充字符（Str）
        print_end   - 结束打印字符（Str）
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (current / float(total)))
    filled_length = int(length * current // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    # 当进度条完成时换行
    if current == total:
        print()
    os.system('cls')
def normalize_bbox(bbox, image_width, image_height):
    """
    将边界框归一化到 [0, 1] 的范围。

    参数:
    bbox (list or tuple): 边界框的坐标，格式为 [x_min, y_min, x_max, y_max]。
    image_width (int or float): 图像的宽度。
    image_height (int or float): 图像的高度。

    返回:
    list: 归一化后的边界框坐标。
    """
    x_min, y_min, x_max, y_max = bbox

    x_min_normalized = x_min / image_width
    y_min_normalized = y_min / image_height
    x_max_normalized = x_max / image_width
    y_max_normalized = y_max / image_height

    return [x_min_normalized, y_min_normalized, x_max_normalized, y_max_normalized]


# 获取训练图片图片id
trains_id_ = os.listdir(train_path)
trains_id = []  # 图像id
for train_id in trains_id_:
    trains_id.append(train_id.split('.')[0])


# 获取测试图片图片id
tests_id_ = os.listdir(test_path)
tests_id = []  # 图像id
for test_id in tests_id_:
    tests_id.append(test_id.split('.')[0])

print('训练ID：',trains_id)
print('测试ID：',tests_id)

# 获取标签
f = open(data_path, 'r')
content = f.read()
annotations = json.loads(content)

# lable_name = []
with open('./dataset/trc.txt','r',encoding='utf-8') as f:
    lable_name = [line.strip() for line in f.readlines()]

for key,value in enumerate(lable_name):
    print(key,value)
width, height = 320,240
owidth,oheight = 2048,2048


def preprocessing(s_id,is_train):

    if is_train is True:
        open_path = train_path
        save_path = "./dataset/train"
    else:
        open_path = test_path
        save_path = "./dataset/val"

    sum_id = s_id
    for id in s_id:
        sum_id.pop(0)
        print_progress_bar(len(sum_id), len(s_id), prefix='Progress:', suffix='Complete', length=50)
        # print(f'处理：{id}中。。。')
        target_dir = annotations['imgs'][f'{id}']
        path = target_dir['path']
        path = path.split('.')[0]
        object = target_dir['objects']
        """
        resize图像
        """

        is_save = False
        encode = 0
        for obj in object:
            encode +=1


            # if object[0]['category'] not in lable_name:
            #     lable_name.append(object[0]['category'])
            xmin = obj['bbox']['xmin']
            ymin = obj['bbox']['ymin']
            # 右下点
            ymax = obj['bbox']['ymax']
            xmax = obj['bbox']['xmax']
            # print('原坐标：',xmin, ymin, xmax, ymax)
            # 计算中心点
            x_center = (xmin + xmax) / 2 + random.randint(-100, 100)
            y_center = (ymin + ymax) / 2 + random.randint(-100, 100)

            # print('原标签框位置：',xmin,ymax,xmax,ymin)

            # 计算图像边缘
            img_xmin = x_center - 160
            img_xmax = x_center + 160

            img_ymin = y_center - 120
            img_ymax = y_center + 120

            new_xmin = abs(xmin -img_xmin)
            new_ymin = abs(ymin -img_ymin)

            new_xmax = abs(new_xmin + (xmax - xmin))
            new_ymax = abs(new_ymin + (ymax - ymin))


            try:
                index = lable_name.index(obj['category'])
                boxx = [new_xmin, new_ymin, new_xmax, new_ymax]
                size = (width, height)
                # x, y, w, h = convert(size, boxx)
                x, y, w, h = normalize_bbox(boxx, size[0], size[1])
                # print('归一化：',x, y, w, h)
                xmin = str(x)
                ymin = str(y)
                xmax = str(w)
                ymax = str(h)
                content = str(index + 1) + ' ' + xmin + ' ' + ymin + ' ' + xmax + ' ' + ymax
                with open(f"{save_path}/{id}_{encode}.txt", 'a') as f:
                    f.write(content + '\n')
            except ValueError:
                print(f"{id}编号的标签为{obj['category']}，列表中没有此标签")
                continue

            try:
                img = Image.open(os.path.join(open_path, f'{id}.jpg'))
                # cropped_img.show()
                cropped_img = img.crop((img_xmin, img_ymin, img_xmax, img_ymax))
                # cropped_img.show()
                # cropped_img.save(f'./dataset/{open_path}/{train_id}.jpg')
                cropped_img.save(os.path.join(save_path, f"{id}_{encode}.jpg"))
                is_save = True
                if is_save is True:
                    # todo：写入文件路径
                    if is_train is True:
                        with open(f'./dataset/train.txt', 'a', encoding='utf-8') as f:
                            f.write(path+ f"_{encode}.jpg" + '\n')
                    else:
                        with open(f'./dataset/val.txt', 'a', encoding='utf-8') as f:
                            f.write(path+ f"_{encode}.jpg" + '\n')


            except FileNotFoundError:
                print(f'{open_path}\{id}.jpg 不存在')

if __name__ == '__main__':
    for id,is_train in (trains_id,True),(tests_id,False):
        preprocessing(id,is_train)

    # print(len(lable_name))
    # with open(".\dataset\data.txt", 'w') as f:
    #     for i in lable_name:
    #         f.write(i + '\n')

```

## 项目代码

能力有限，并未对[Yolo-FastestV2](https://github.com/dog-qiuqiu/Yolo-FastestV2)做很大的改动，只是对该项目在本机中运行做遇到问题，做了一点改动。代码也上传至我的github。

## 训练结果

这样的结果其实我也很惊讶，即使是单目标检测，结果精度都这么低，相比较于coco数据集精度差了将近一倍，我暂时觉得是因为**目标太小**的问题。我暂时无法解决，我看到[Yolo-FastestV2](https://github.com/dog-qiuqiu/Yolo-FastestV2)的作者有采用FPN模型，但是精度还是很低，其实还可以尝试采用其他模型作为训练，但我毕竟不是CV方向，并且时间不够，就没有继续往下钻研。我觉得目标小，有点类似于遥感的目标检测，可以采用遥感的目标检测算法，来检测该数据集（如过不是在嵌入式单片机上跑的话）可能会有好的结果出来。

```shell
D:\anaconda\envs\gd32ai\lib\site-packages\torch\functional.py:512: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at C:\cb\pytorch_1000000000000\work\aten\src\ATen\native\TensorShape.cpp:3588.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
Evaluation model:: 100%|██████████| 60/60 [00:38<00:00,  1.54it/s]
computer PR...
Evaluation model:: 100%|██████████| 60/60 [00:10<00:00,  5.74it/s]
Precision:0.281061 Recall:0.445965 AP:0.318521 F1:0.337140
```



# :zap:Yolo-FastestV2:zap:[![DOI](https://zenodo.org/badge/386585431.svg)](https://zenodo.org/badge/latestdoi/386585431)
![image](https://github.com/dog-qiuqiu/Yolo-FastestV2/blob/main/img/demo.png)
* ***Simple, fast, compact, easy to transplant***
* ***Less resource occupation, excellent single-core performance, lower power consumption***
* ***Faster and smaller:Trade 0.3% loss of accuracy for 30% increase in inference speed, reducing the amount of parameters by 25%***
* ***Fast training speed, low computing power requirements, training only requires 3GB video memory, gtx1660ti training COCO 1 epoch only takes 4 minutes***
* ***算法介绍：https://zhuanlan.zhihu.com/p/400474142 交流qq群:1062122604***
# Evaluating indicator/Benchmark
Network|COCO mAP(0.5)|Resolution|Run Time(4xCore)|Run Time(1xCore)|FLOPs(G)|Params(M)
:---:|:---:|:---:|:---:|:---:|:---:|:---:
[Yolo-FastestV2](https://github.com/dog-qiuqiu/Yolo-FastestV2/tree/main/modelzoo)|24.10 %|352X352|3.29 ms|5.37 ms|0.212|0.25M
[Yolo-FastestV1.1](https://github.com/dog-qiuqiu/Yolo-Fastest/tree/master/ModelZoo/yolo-fastest-1.1_coco)|24.40 %|320X320|4.23 ms|7.54 ms|0.252|0.35M
[Yolov4-Tiny](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg)|40.2%|416X416|26.00ms|55.44ms|6.9|5.77M

* ***Test platform Mate 30 Kirin 990 CPU，Based on [NCNN](https://github.com/Tencent/ncnn)***
# Improvement
* Different loss weights for different scale output layers
* The backbone is replaced with a more lightweight shufflenetV2
* Anchor matching mechanism and loss are replaced by YoloV5, and the classification loss is replaced by softmax cross entropy from sigmoid
* Decouple the detection head, distinguish obj (foreground background classification), cls (category classification), reg (detection frame regression) 3 branches,  
# How to use
## Dependent installation
  * PIP
  ```
  pip3 install -r requirements.txt
  ```
## Test
* Picture test
  ```
  python3 test.py --data data/coco.data --weights modelzoo/coco2017-0.241078ap-model.pth --img img/000139.jpg
  ```
<div align=center>
<img src="https://github.com/dog-qiuqiu/Yolo-FastestV2/blob/main/img/000139_result.png"> />
</div>

## How to train
### Building data sets(The dataset is constructed in the same way as darknet yolo)
* The format of the data set is the same as that of Darknet Yolo, Each image corresponds to a .txt label file. The label format is also based on Darknet Yolo's data set label format: "category cx cy wh", where category is the category subscript, cx, cy are the coordinates of the center point of the normalized label box, and w, h are the normalized label box The width and height, .txt label file content example as follows:
  ```
  11 0.344192634561 0.611 0.416430594901 0.262
  14 0.509915014164 0.51 0.974504249292 0.972
  ```
* The image and its corresponding label file have the same name and are stored in the same directory. The data file structure is as follows:
  ```
  .
  ├── train
  │   ├── 000001.jpg
  │   ├── 000001.txt
  │   ├── 000002.jpg
  │   ├── 000002.txt
  │   ├── 000003.jpg
  │   └── 000003.txt
  └── val
      ├── 000043.jpg
      ├── 000043.txt
      ├── 000057.jpg
      ├── 000057.txt
      ├── 000070.jpg
      └── 000070.txt
  ```
* Generate a dataset path .txt file, the example content is as follows：
  
  train.txt
  ```
  /home/qiuqiu/Desktop/dataset/train/000001.jpg
  /home/qiuqiu/Desktop/dataset/train/000002.jpg
  /home/qiuqiu/Desktop/dataset/train/000003.jpg
  ```
  val.txt
  ```
  /home/qiuqiu/Desktop/dataset/val/000070.jpg
  /home/qiuqiu/Desktop/dataset/val/000043.jpg
  /home/qiuqiu/Desktop/dataset/val/000057.jpg
  ```
* Generate the .names category label file, the sample content is as follows:

  category.names
  ```
  person
  bicycle
  car
  motorbike
  ...
  
  ```
* The directory structure of the finally constructed training data set is as follows:
  ```
  .
  ├── category.names        # .names category label file
  ├── train                 # train dataset
  │   ├── 000001.jpg
  │   ├── 000001.txt
  │   ├── 000002.jpg
  │   ├── 000002.txt
  │   ├── 000003.jpg
  │   └── 000003.txt
  ├── train.txt              # train dataset path .txt file
  ├── val                    # val dataset
  │   ├── 000043.jpg
  │   ├── 000043.txt
  │   ├── 000057.jpg
  │   ├── 000057.txt
  │   ├── 000070.jpg
  │   └── 000070.txt
  └── val.txt                # val dataset path .txt file

  ```
### Get anchor bias
* Generate anchor based on current dataset
  ```
  python3 genanchors.py --traintxt ./train.txt
  ```
* The anchors6.txt file will be generated in the current directory,the sample content of the anchors6.txt is as follows:
  ```
  12.64,19.39, 37.88,51.48, 55.71,138.31, 126.91,78.23, 131.57,214.55, 279.92,258.87  # anchor bias
  0.636158                                                                             # iou
  ```
### Build the training .data configuration file
* Reference./data/coco.data
  ```
  [name]
  model_name=coco           # model name

  [train-configure]
  epochs=300                # train epichs
  steps=150,250             # Declining learning rate steps
  batch_size=64             # batch size
  subdivisions=1            # Same as the subdivisions of the darknet cfg file
  learning_rate=0.001       # learning rate

  [model-configure]
  pre_weights=None          # The path to load the model, if it is none, then restart the training
  classes=80                # Number of detection categories
  width=352                 # The width of the model input image
  height=352                # The height of the model input image
  anchor_num=3              # anchor num
  anchors=12.64,19.39, 37.88,51.48, 55.71,138.31, 126.91,78.23, 131.57,214.55, 279.92,258.87 #anchor bias

  [data-configure]
  train=/media/qiuqiu/D/coco/train2017.txt   # train dataset path .txt file
  val=/media/qiuqiu/D/coco/val2017.txt       # val dataset path .txt file 
  names=./data/coco.names                    # .names category label file
  ```
### Train
* Perform training tasks
  ```
  python3 train.py --data data/coco.data
  ```
### Evaluation
* Calculate map evaluation
  ```
  python3 evaluation.py --data data/coco.data --weights modelzoo/coco2017-0.241078ap-model.pth
  ```
# Deploy
## NCNN
* Convert onnx
  ```
  python3 pytorch2onnx.py --data data/coco.data --weights modelzoo/coco2017-0.241078ap-model.pth --output yolo-fastestv2.onnx
  ```
* onnx-sim
  ```
  python3 -m onnxsim yolo-fastestv2.onnx yolo-fastestv2-opt.onnx
  ```
* Build NCNN
  ```
  git clone https://github.com/Tencent/ncnn.git
  cd ncnn
  mkdir build
  cd build
  cmake ..
  make
  make install
  cp -rf ./ncnn/build/install/* ~/Yolo-FastestV2/sample/ncnn
  ```
* Covert ncnn param and bin
  ```
  cd ncnn/build/tools/onnx
  ./onnx2ncnn yolo-fastestv2-opt.onnx yolo-fastestv2.param yolo-fastestv2.bin
  cp yolo-fastestv2* ../
  cd ../
  ./ncnnoptimize yolo-fastestv2.param yolo-fastestv2.bin yolo-fastestv2-opt.param yolo-fastestv2-opt.bin 1
  cp yolo-fastestv2-opt* ~/Yolo-FastestV2/sample/ncnn/model
  ```
* run sample
  ```
  cd ~/Yolo-FastestV2/sample/ncnn
  sh build.sh
  ./demo
  ```
# Reference
* https://github.com/Tencent/ncnn
* https://github.com/AlexeyAB/darknet
* https://github.com/ultralytics/yolov5
* https://github.com/eriklindernoren/PyTorch-YOLOv3
