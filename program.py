from collections import deque
from ultralytics import YOLO
from ppvideo import PaddleVideo
import cv2
import numpy as np
import time
import os
import math
import argparse
import imageio


# 新增ONNX推理类
class PPTSM_ONNX_Inference:
    def __init__(self, onnx_path='ppTSM.onnx'):
        import onnxruntime as ort
        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, frame_buffer):
        # 转换为模型需要的输入格式 [batch, num_seg, C, H, W]
        processed = np.stack([
            cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (320, 320))
            for f in frame_buffer
        ], axis=0).astype(np.float32) / 255.0
        return processed.transpose(0, 3, 1, 2)[None, :]  # 添加batch维度

    def predict(self, frame_buffer):
        input_data = self.preprocess(frame_buffer)
        outputs = self.session.run([self.output_name], {self.input_name: input_data})

        # # 调试信息
        # print("实际输出形状:", outputs[0].shape)  # 打印输出张量维度
        # print("输出样例值:", outputs[0][0])  # 打印第一个样本的输出

        return {'fight': outputs[0][0][1]}  # 输出形状为(1,2)


def test_video(input_path, yolo_path, pptsm_path):
    # 实例化检测和分类模型
    yolo_model = YOLO(yolo_path)
    pptsm_model = PPTSM_ONNX_Inference(onnx_path=pptsm_path)

    '''参数记录部分'''
    deque_maxlen = 8  # 缓存区最大长度
    frame_interval = 0.1  # 以多少秒的间隔进入缓存区
    show_time = 2  # 标注持续多长时间
    det_conf = 0.5  # 检测(YOLO)置信度
    det_iou = 0.6  # 检测(YOLO)IOU阈值
    cls_conf = 0.5  # 分类(PP-TSM)置信度
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码器

    '''标记部分'''
    last_add_time = 0  # 新增“缓存”的时间戳记录
    last_fight_time = 0  # 新增“标识”的时间戳记录

    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # 初始化结果视频写入器
    writer = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
    # 初始化缓存区
    frame_buffer = deque(maxlen=deque_maxlen)

    while cap.isOpened():
        # 开始读取帧
        ret, frame = cap.read()
        if not ret: break

        # 获取当前时间
        current_time = time.time()
        # 只有时间间隔超过xx秒时才添加新帧至队列中（模拟抽帧）
        if current_time - last_add_time >= frame_interval:
            frame_buffer.append(frame.copy())
            last_add_time = current_time  # 更新最后添加时间

        ''' YOLOv8一级检测（使用视频取帧）'''
        detections = yolo_model.predict(source=frame, device="0",
                                        imgsz=640, conf=det_conf, iou=det_iou, line_width=1)[0]  # box.cls若为0，则为abuse

        # if (any(int(box.cls.item()) == 0 for box in detections.boxes)
        #         and (time.time() - last_fight_time > 0.5)):           # 1.设置0.5秒的懒惰允许

        if any(int(box.cls.item()) == 0 for box in detections.boxes):  # 2.默认设置,凡是检测到就来到分类

            if len(frame_buffer) < deque_maxlen:  # 若缓存未满则以yolo输出为准
                last_fight_time = time.time()
                print("Detect without cls!")

            ''' PP-TSM二级检测（使用缓存帧）'''
            if len(frame_buffer) == deque_maxlen:  # *** 这里需要确保缓存已满
                result = pptsm_model.predict(list(frame_buffer))
                print("result is :", result)
                if abs(result['fight']) > cls_conf:  # 不太理解输出为什么是[正值,负值],这里直接取绝对值处理
                    print("Detect with cls!")
                    last_fight_time = time.time()

        '''  标注分析：
        (前两秒，队列还未满时)若yolo检测出时，直接会显示"FIGHT!"
        (之后)每当两次检测均为fight时，更新fight_time，两秒内会一直显示"FIGHT!"   '''
        # 动态标注
        if time.time() - last_fight_time < show_time:
            cv2.putText(frame, "FIGHT!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX,
                        3, (0, 0, 255), 5)

        # 写入输出
        writer.write(frame)

    cap.release()
    writer.release()


if __name__ == "__main__":

    # ========模型路径========
    yolo_pt_path = "/civi/data/full_dataset/runs/detect/train_full-and-ours_more/weights/best.pt"
    pptsm_onnx_path = "./inference/417ppTSM_finetune/ppTSM.onnx"

    # 检查模型文件是否存在
    if not os.path.exists(yolo_pt_path):
        print(f"YOLOv8模型文件不存在: {yolo_pt_path}")
    if not os.path.exists(pptsm_onnx_path):
        print(f"PP-TSM模型文件不存在: {pptsm_onnx_path}")

    # ========视频路径========
    # input_video = "test_ours_more/fight_3_2.avi"
    input_video = "dataset/Our-more-clips_two-type/fight/fight_4_1_008.mp4"

    # ========开始检测========
    test_video(input_video, yolo_pt_path, pptsm_onnx_path)
