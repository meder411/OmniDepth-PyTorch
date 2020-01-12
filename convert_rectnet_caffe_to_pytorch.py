import h5py
import argparse
import torch

# Parse arguments
parser = argparse.ArgumentParser(description='Convert Caffe model to PyTorch')
parser.add_argument('-i', '--input', required=True, help='Path to input file')
parser.add_argument('-o', '--output', required=True, help='Path to output file')

caffe_model_path = args.input
pytorch_model_path = args.output

# Associate layer names from the caffe network to names of the PyTorch network
rectnet_name_dict = {
    'dilBlock01_1x1': 'encoder1_3',
    'dilBlock01_conv01': 'encoder1_1',
    'dilBlock01_conv02': 'encoder1_2',
    'dilBlock01_strConv01': 'encoder1_0',
    'dilBlock02_1x1': 'encoder2_2',
    'dilBlock02_conv01': 'encoder2_0',
    'dilBlock02_conv02': 'encoder2_1',
    'downBlock01_conv01': 'encoder0_0',
    'downBlock01_conv02': 'encoder0_1',
    'downBlock01_conv03': 'encoder0_2',
    'prediction_x_1': 'prediction1',
    'prediction_x_1_1x1': 'decoder1_2',
    'prediction_x_2': 'prediction0',
    'preprocBlock01_central_conv': 'input0_3',
    'preprocBlock01_dist01_conv': 'input0_0',
    'preprocBlock01_dist02_conv': 'input0_1',
    'preprocBlock01_dist03_conv': 'input0_2',
    'preprocBlock02_central_conv': 'input1_3',
    'preprocBlock02_dist01_conv': 'input1_0',
    'preprocBlock02_dist02_conv': 'input1_1',
    'preprocBlock02_dist03_conv': 'input1_2',
    'upBlock01_conv': 'decoder0_1',
    'upBlock01_deconv': 'decoder0_0',
    'upBlock02_conv': 'decoder1_1',
    'upBlock02_deconv': 'decoder1_0'
}

# Load the caffe model
caffe_data = h5py.File(caffe_model_path, 'r')

# Find the group name storing all the weights
group = list(caffe_data.keys())[0]

# Get the layer names
layer_data = caffe_data[group]
layer_keys = list(caffe_data[group].keys())

pytorch_state_dict = {}

# Go through all layers
for layer_name in layer_keys:

    # If it's a network layer (i.e. not part of the loss or preprocessing)
    if len(list(layer_data[layer_name].keys())) > 0:
        if layer_name in rectnet_name_dict:

            # Prefix `module` because it gets run in DataParallel
            pytorch_name = 'module.' + rectnet_name_dict[layer_name]

            # There's no submodule for predictions, but others will have .conv
            if (layer_name != 'prediction_x_1') and (layer_name !=
                                                     'prediction_x_2'):
                pytorch_name += '.conv'

            # Go through both weights and biases
            for k in layer_data[layer_name].keys():
                sub_layer_name = layer_name + '_' + k

                # Weights
                if k == '0':
                    pytorch_state_dict[pytorch_name
                                       + '.weight'] = torch.from_numpy(
                                           layer_data[layer_name][k].value)
                # Biases
                elif k == '1':
                    pytorch_state_dict[pytorch_name
                                       + '.bias'] = torch.from_numpy(
                                           layer_data[layer_name][k].value)

for k in pytorch_state_dict.keys():
    # This flips the input channels because the original paper used OpenCV's BGR format
    if ('input0' in k) and ('weight' in k):
        pytorch_state_dict[k] = torch.flip(pytorch_state_dict[k], (1, ))
    print(k, ':', pytorch_state_dict[k].shape)

# Save the model as a checkpoint
checkpoint = dict()
checkpoint['epoch'] = 0
checkpoint['state_dict'] = pytorch_state_dict
torch.save(checkpoint, pytorch_model_path)