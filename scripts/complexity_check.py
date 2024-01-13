import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(script_dir, '..')
sys.path.append(root_dir)

import datetime
import tensorflow as tf
from model.arch import LYT, Denoiser
import argparse
import logging
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel(logging.ERROR)

def get_time():
    current_time = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    return current_time

def get_total_params(model):
    total_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    total_non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    return total_params, total_non_trainable_params

def compute_flops(model, input_shape=(256,256,3)):
    input_layer = tf.keras.Input(shape=input_shape)
    
    generated_images = model(input_layer)
    
    model_ = tf.keras.Model(inputs=input_layer, outputs=generated_images)
    
    def get_flops(model):
        concrete = tf.function(lambda inputs: model(inputs))
        concrete_func = concrete.get_concrete_function(
            [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
    
        frozen_func, _ = convert_variables_to_constants_v2_as_graph(concrete_func)
        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(frozen_func.graph.as_graph_def(), name='')
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            
            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
    
            return flops.total_float_ops
    
    total_flops = get_flops(model_)
    return total_flops

def compute_complexity(shape=(256,256,3)):
    print('LYT-Net 2024 (c) Brateanu, A., Balmez, R., Avram A., Orhei, C.C.')
    print(f"({get_time()}) Computing complexity of LYT-Net.")

    # Build model
    denoiser_cb = Denoiser(16)
    denoiser_cr = Denoiser(16)
    denoiser_cb.build(input_shape=(None,None,None,1))
    denoiser_cr.build(input_shape=(None,None,None,1))
    model = LYT(filters=32, denoiser_cb=denoiser_cb, denoiser_cr=denoiser_cr)
    model.build(input_shape=(None,None,None,3))

    # Get stats
    flops = compute_flops(model, shape)
    param_count, _ = get_total_params(model)
    
    print(f"({get_time()}) LYT-Net complexity:")
    print(f'FLOPs: {(flops / (1024*1024*1024)):.2f} G')
    print(f'Params: {param_count}')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Complexity check script')
    parser.add_argument('--shape', type=str, required=False, help='Input shape')
    args = parser.parse_args()
    if args.shape:
        shape = args.shape[1:len(args.shape)-1]
        shape = shape.split(',')
        shape = [int(x) for x in shape]
        compute_complexity(shape)  
    else:
        compute_complexity()
