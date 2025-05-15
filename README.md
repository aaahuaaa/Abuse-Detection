# Abuse Detection Project
## 数据集
在[百度网盘](https://pan.baidu.com/s/1lMUZjxLii649VWwd1HdKLg?pwd=1234)中下载，密码：1234 <br>
- 【公共数据集pp_five_datasets】<br>
  - RWF-2000：最好，最适配，既有室内室外，也有双人多人，可惜小孩的较少<br>
  - CCTV-Fight：来源基本都是TV的新闻报道类，较长，很难用好<br>
  - SCFD：时长基本2s左右，fight仅包含打架部分，挺多是多人间互动<br>
  - RLVSD：1k个Violence和1k个nonViolence视频<br>
  - DAVDV：室内模拟，双镜头，均为5s左右<br>
  - HFDD：冰球比赛视频，长度均为1s<br>
- 【自制数据集】<br>
  - 网上报道的截图：200张左右<br>
  - 办公室模拟场景：20段视频，抽帧2000张左右<br>

## 模型权重
在[百度网盘](https://pan.baidu.com/s/1UusJyYP9uLINSWgU8LImlw?pwd=1234)中下载，密码：1234 <br>

## 模型一：YOLOv8一级检测
### 模型训练
于"yolo/train/images/"  中加入新的图像数据 <br>
于"yolo/train/labels/"  中加入新的标签数据 <br>
```
cd /your/project/dir  # 进入YOLOv8的项目目录
python train.py
```
训练之后进入"~/runs/detect/"中，找到对应train文件夹及其best.pt
``` 
(test.py) model = YOLO('~/runs/detect/xxx(your_train_project)/weights/best.pt')
```
``` 
(get onnx.py) model = YOLO("~/runs/detect/xxx(your_train_project)/weights/best.pt")
```
之后进行测试：
```
python test.py
# 若需要转成onnx文件
python get_onnx.py
```
## 模型二：ppTSM二级检测
参考[官方文档](https://github.com/PaddlePaddle/PaddleVideo/tree/develop/applications/FightRecognition) <br>
### 模型训练
1. 视频抽帧 <br>
为了加快训练速度，将视频进行抽帧。
```
cd /your/project/dir  # 进入ppTSM的项目目录
python data/ucf101/extract_rawframes.py (your_data_dir) (store_dir) --level 2 --ext mp4
```
2. 训练集和验证集划分
```
python SplitDatasetToList.py
```
生成 fight_train_list.txt 和 fight_val_list.txt <br>

3. 开始训练
```
export CUDA_VISIBLE_DEVICES=0,1,2

python -B -m paddle.distributed.launch --gpus="0,1,2" \
   --log_dir=(your_log_file)  main.py  --validate \
   -c pptsm_fight_frames_dense.yaml
```
4. 导出模型 
`需要确保yaml文件中132行 model_name为 ppTSM`
```
python tools/export_model.py \
-c /civi/data/ppTSM/pptsm_fight_frames_dense.yaml\
-p /civi/data/ppTSM/output/.../ppTSM_best.pdparams\
-o /civi/data/ppTSM/inference/...
```
5. 执行推理
```
python tools/predict.py \
--input_file dataset/.../nofight.mp4 \
--config pptsm_fight_frames_dense.yaml \
--model_file inference/.../ppTSM.pdmodel \
--params_file inference/.../ppTSM.pdiparams \
--use_gpu=True \
--use_tensorrt=False
```
`注意，过程中如果涉及ava_predict导入报错，".ava_predict"和"ava_predict"互相改就好了`<br>

6. 转onnx
```
paddle2onnx \
--model_dir=./inference/... \
--model_filename=ppTSM.pdmodel \
--params_filename=ppTSM.pdiparams \
--save_file=./inference/.../ppTSM.onnx \
--opset_version=10 \
--enable_onnx_checker=True
```
7. onnx预测推理
```
python deploy/paddle2onnx/predict_onnx.py \
--input_file dataset/.../nofight.mp4 \
--config pptsm_fight_frames_dense.yaml \
--onnx_file=./inference/.../ppTSM.onnx
```
### 直接推理
`（目前仅对离线视频）`
```
conda activate py38
cd /civi/data/ppTSM/
python program.py
```
program.py 文件路径修改：
```
yolo_pt_path = "/civi/data/full_dataset/.../best.pt" # yolo的.pt文件
pptsm_onnx_path = "./inference/.../ppTSM.onnx"       # ppTSM的.onnx文件
input_video = ".../fight.mp4"                        # 要推理的视频文件
```
program.py 提供可修改参数：
```
deque_maxlen = 8                          # 缓存区最大长度
frame_interval = 0.1                      # 以多少秒的间隔进入缓存区
show_time = 2                             # 标注持续多长时间
det_conf = 0.5                            # 检测(YOLO)置信度
det_iou = 0.6                             # 检测(YOLO)IOU阈值
cls_conf = 0.5                            # 分类(PP-TSM)置信度
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码器
```
