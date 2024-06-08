# -*- coding: UTF-8 -*-

# @Project ：Yolo-FastestV2 
# @File    ：pytorch2tflite.py
# @IDE     ：PyCharm 
# @Author  ：liguochun0304@163.com
# @Date    ：2024/6/3 10:38


import argparse

import torch
import model.detector
import utils.utils
import tensorflow as tf

import onnx
from onnx_tf.backend import prepare
if __name__ == '__main__':
    # 指定训练配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/trc.data',
                        help='Specify training profile *.data')
    parser.add_argument('--weights_0603', type=str,
                        default='D:\git_km\Yolo-FastestV2\weights/traffic-200-epoch-0.202316ap-model.pth',
                        help='The path of the .pth model to be transformed')

    parser.add_argument('--output', type=str, default='./model.onnx',
                        help='The path where the onnx model is saved')

    opt = parser.parse_args()
    cfg = utils.utils.load_datafile(opt.data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.detector.Detector(cfg["classes"], cfg["anchor_num"], True, True).to(device)
    model.load_state_dict(torch.load(opt.weights, map_location=device))
    # sets the module in eval node
    model.eval()

    test_data = torch.rand(1, 3, cfg["height"], cfg["width"]).to(device)
    torch.onnx.export(model,  # model being run
                      test_data,  # model input (or a tuple for multiple inputs)
                      opt.output,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights_0603 inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True)  # whether to execute constant folding for optimization


    onnx_model = onnx.load('model.onnx')
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph('model_tf')

    # 加载TensorFlow模型
    converter = tf.lite.TFLiteConverter.from_saved_model('model_tf')
    tflite_model = converter.convert()

    # 将TFLite模型保存为文件
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)


