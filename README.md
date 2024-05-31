## README: 在VOC数据集上训练和测试Faster R-CNN和YOLO V3

### 前置条件

1. **Python 3.7+**
2. **CUDA 10.1+（用于GPU训练）**
3. **PyTorch 1.6+**
4. **mmdetection框架**

### 安装

#### MMDetection

1. 克隆MMDetection库并进入目录：

```bash
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
```

2. 安装依赖并编译：

```bash
pip install -r requirements/build.txt
pip install -v -e .
```

### 数据准备

1. 下载并解压VOC 2012数据集：

```bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar
```

2. 确保VOC数据集文件结构如下：

```
VOCdevkit/
└── VOC2012/
    ├── Annotations/
    ├── ImageSets/
    ├── JPEGImages/
    └── ...
```

### 模型训练与测试

#### 使用MMDetection训练和测试Faster R-CNN

1. 配置文件修改：
   
   在`configs/faster_rcnn`目录下找到适合VOC数据集的配置文件，例如`faster_rcnn_r50_fpn_1x_voc0712.py`。将其复制并重命名为`faster_rcnn_r50_fpn_1x_voc2012.py`，并修改其中的数据路径以适应VOC 2012数据集。

2. 修改数据路径：

```python
data = dict(
    train=dict(
        type='VOCDataset',
        ann_file='VOCdevkit/VOC2012/ImageSets/Main/trainval.txt',
        img_prefix='VOCdevkit/VOC2012/'),
    val=dict(
        type='VOCDataset',
        ann_file='VOCdevkit/VOC2012/ImageSets/Main/val.txt',
        img_prefix='VOCdevkit/VOC2012/'),
    test=dict(
        type='VOCDataset',
        ann_file='VOCdevkit/VOC2012/ImageSets/Main/test.txt',
        img_prefix='VOCdevkit/VOC2012/'))
```

3. 训练模型：

```bash
python tools/train.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc2012.py
```

4. 测试模型：

```bash
python tools/test.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc2012.py checkpoints/latest.pth --eval mAP
```

#### 使用MMDetection训练和测试YOLO V3

1. 配置文件修改：
   
   在`configs/yolo`目录下找到适合VOC数据集的配置文件，例如`yolov3_d53_mstrain-608_273e_voc0712.py`。将其复制并重命名为`yolov3_d53_mstrain-608_273e_voc2012.py`，并修改其中的数据路径以适应VOC 2012数据集。

2. 修改数据路径：

```python
data = dict(
    train=dict(
        type='VOCDataset',
        ann_file='VOCdevkit/VOC2012/ImageSets/Main/trainval.txt',
        img_prefix='VOCdevkit/VOC2012/'),
    val=dict(
        type='VOCDataset',
        ann_file='VOCdevkit/VOC2012/ImageSets/Main/val.txt',
        img_prefix='VOCdevkit/VOC2012/'),
    test=dict(
        type='VOCDataset',
        ann_file='VOCdevkit/VOC2012/ImageSets/Main/test.txt',
        img_prefix='VOCdevkit/VOC2012/'))
```

3. 训练模型：

```bash
python tools/train.py configs/yolo/yolov3_d53_mstrain-608_273e_voc2012.py
```

4. 测试模型：

```bash
python tools/test.py configs/yolo/yolov3_d53_mstrain-608_273e_voc2012.py checkpoints/latest.pth --eval mAP
```

### 结果可视化

#### 对比Faster R-CNN第一阶段proposal box和最终预测结果

1. 在测试集上选取4张图像，并使用训练好的Faster R-CNN模型进行预测：

```bash
python tools/test.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc2012.py checkpoints/latest.pth --show-dir results
```

2. 将生成的proposal box和最终预测结果进行对比，可视化保存结果。

#### 对比不同模型在非VOC图像上的检测结果

1. 准备3张不在VOC数据集中的图片，并进行预测：

```bash
python tools/test.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc2012.py checkpoints/latest.pth --show-dir results_non_voc
python tools/test.py configs/yolo/yolov3_d53_mstrain-608_273e_voc2012.py checkpoints/latest.pth --show-dir results_non_voc
```

2. 可视化并比较两个模型在这三张图片上的检测结果（包括bounding box、类别标签和得分）。
