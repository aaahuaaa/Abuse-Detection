# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from os import path as osp
import paddle
from paddle import inference
from paddle.inference import Config, create_predictor
from utils import build_inference_helper
from paddlevideo.utils import get_config

def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "1")

    # general params
    parser = argparse.ArgumentParser("PaddleVideo Inference model script")
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default='pptsm_fight_frames_dense.yaml')
    parser.add_argument(
        '-o',
        '--override',
        action='append',
        default=[],
        help='config options to be overridden')
    parser.add_argument("-i", "--input_file", type=str, help="input file path")
    parser.add_argument(
        "--time_test_file",
        type=str2bool,
        default=False,
        help="whether input time test file")
    parser.add_argument("--model_file", type=str)
    parser.add_argument("--params_file", type=str)

    # params for paddle predict
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--use_xpu", type=str2bool, default=False)
    parser.add_argument("--use_npu", type=str2bool, default=False)
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--gpu_mem", type=int, default=8000)
    parser.add_argument("--enable_benchmark", type=str2bool, default=False)
    parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
    parser.add_argument("--cpu_threads", type=int, default=None)
    parser.add_argument("--disable_glog", type=str2bool, default=False)
    # parser.add_argument("--hubserving", type=str2bool, default=False)  #TODO

    return parser.parse_args()


def create_paddle_predictor(args, cfg):
    config = Config(args.model_file, args.params_file)
    if args.use_gpu:
        config.enable_use_gpu(args.gpu_mem, 0)
    else:
        config.disable_gpu()
        if args.cpu_threads:
            config.set_cpu_math_library_num_threads(args.cpu_threads)

    # config.disable_glog_info()
    config.switch_ir_optim(args.ir_optim)  # default true
    if args.use_tensorrt:

        precision = inference.PrecisionType.Float32

        # calculate real max batch size during inference when tenrotRT enabled
        max_batch_size = args.batch_size
        if 'num_seg' in cfg.INFERENCE:
            ''' 
            num_seg: number of segments when extracting frames.
            seg_len: number of frames extracted within a segment, default to 1.
            num_views: the number of video frame groups obtained by cropping and flipping,
            uniformcrop=3, tencrop=10, centercrop=1.
            '''
            num_seg = cfg.INFERENCE.num_seg
            seg_len = cfg.INFERENCE.get('seg_len', 1)
            num_views = 1
            if 'tsm' in cfg.model_name.lower():
                num_views = 1  # CenterCrop
            max_batch_size = args.batch_size * num_views * num_seg * seg_len
        config.enable_tensorrt_engine(
            precision_mode=precision, max_batch_size=max_batch_size)
    config.enable_memory_optim()

    # use zero copy
    config.switch_use_feed_fetch_ops(False)

    # disable glog
    if args.disable_glog:
        config.disable_glog_info()

    # for ST-GCN tensorRT case usage
    # config.delete_pass("shuffle_channel_detect_pass")

    predictor = create_predictor(config)

    return config, predictor


def parse_file_paths(input_path: str) -> list:
    if osp.isfile(input_path):
        files = [
            input_path,
        ]
    else:
        files = os.listdir(input_path)
        files = [
            file for file in files
            if (file.endswith(".avi") or file.endswith(".mp4"))
        ]
        files = [osp.join(input_path, file) for file in files]
    return files


def main():
    """predict using paddle inference model"""

    args = parse_args()
    cfg = get_config(args.config, overrides=args.override, show=False)

    model_name = cfg.model_name
    print(f"Inference model({model_name})...")
    InferenceHelper = build_inference_helper(cfg.INFERENCE)
    inference_config, predictor = create_paddle_predictor(args, cfg)

    # get input_tensor and output_tensor
    input_names = predictor.get_input_names()
    output_names = predictor.get_output_names()
    input_tensor_list = []
    output_tensor_list = []
    for item in input_names:
        input_tensor_list.append(predictor.get_input_handle(item))
    for item in output_names:
        output_tensor_list.append(predictor.get_output_handle(item))

    files = parse_file_paths(args.input_file)

    # Inferencing process
    batch_num = args.batch_size
    for st_idx in range(0, len(files), batch_num):
        ed_idx = min(st_idx + batch_num, len(files))

        # Pre process batched input
        batched_inputs = InferenceHelper.preprocess_batch(
            files[st_idx:ed_idx])

        # run inference
        for i in range(len(input_tensor_list)):
            input_tensor_list[i].copy_from_cpu(batched_inputs[i])
        predictor.run()

        batched_outputs = []
        for j in range(len(output_tensor_list)):
            batched_outputs.append(output_tensor_list[j].copy_to_cpu())

        InferenceHelper.postprocess(batched_outputs,
                                    not args.enable_benchmark)


if __name__ == "__main__":
    main()
