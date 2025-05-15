from ppvideo import PaddleVideo
import cv2
import numpy as np
import paddle

input_path = 'fight.avi'
pptsm_model = PaddleVideo(model_name='ppTSM', use_gpu=False)
result = pptsm_model.predict(input_path)