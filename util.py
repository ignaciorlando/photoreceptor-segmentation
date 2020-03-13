
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from matplotlib import pyplot as plt

from abc import ABC, abstractmethod


def bayesian_eval_mode(module):
    module.train()
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()



class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)



def get_final_activation_unit(activation_name):

    if activation_name == 'Softmax2d':
        return nn.Softmax2d()
    elif activation_name == 'Sigmoid':
        return nn.Sigmoid()
    elif activation_name == 'Softplus':
        return nn.Softplus()
    else:
        return None



def get_activation_unit(unit_name):
    '''
    Given the name of an activation unit, return it
    '''

    if unit_name == 'relu':
        # Basic Rectified Linear Unit
        return nn.ReLU()

    elif unit_name == 'prelu':
        # Parametric Rectified Linear Unit
        # from the paper: Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
        # (https://arxiv.org/pdf/1502.01852)
        return nn.PReLU()

    elif unit_name == 'elu':
        # Exponential Linear Unit
        # from the paper: Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
        # (https://arxiv.org/pdf/1511.07289.pdf)
        return nn.ELU()

    elif unit_name == 'lrelu':
        # Leaky ReLU
        # Better described here: https://arxiv.org/pdf/1505.00853.pdf
        return nn.LeakyReLU()

    elif unit_name == 'rrelu':
        # Randomized Leaky Rectified Linear Unit
        # from the paper: Empirical Evaluation of Rectified Activations in Convolutional Network
        # (https://arxiv.org/pdf/1505.00853.pdf)
        return nn.RReLU()

    elif unit_name == 'selu':
        # Scale Exponential Linear Unit
        # from the paper: Self-Normalizing Neural Networks 
        # (https://arxiv.org/pdf/1505.00853.pdf)
        return nn.SELU()

    else:
        raise ValueError('Activation unit {} unknown'.format(unit_name))



def get_pooling_layer(pooling_name, kernel_size):
    '''
    Given the name of a pooling layer, returns its corresponding layer
    '''

    if pooling_name == 'max':
        # Max Pooling layer
        return nn.MaxPool2d(kernel_size=kernel_size)

    elif pooling_name == 'avg':
        # Avg Pooling layer
        return nn.AvgPool2d(kernel_size=kernel_size)

    else:
        raise ValueError('Pooling layer {} unknown'.format(pooling_name))



def get_dropout(dropout_type, keep_probability):
    '''
    Returns a dropout strategy
    '''

    if dropout_type == 'standard':
        # Standard dropout
        # (https://arxiv.org/abs/1207.0580)
        return nn.Dropout(keep_probability)

    elif dropout_type == 'spatial':
        # Spatial dropout (Dropout2D in pytorch)
        # (http://arxiv.org/abs/1411.4280)
        return nn.Dropout2d(keep_probability)

    else:
        raise ValueError('Dropout type {} unknown'.format(dropout_type))



def get_normalization_strategy(config):
    '''
    Return the normalization approach
    '''

    if 'normalization' in config['training']:
        normalization_status_name = config['training']['normalization']
    else:
        normalization_status_name = 'z-score'

    if normalization_status_name == 'z-score':
        # normalize to zero mean and unit variance
        return Z_Normalization()

    elif normalization_status_name == 'maximum':
        # normalize with maximum value
        return MaximumNormalization()

    elif normalization_status_name == 'range':
        # normalize with range
        return RangeNormalization()

    elif normalization_status_name == 'constant':
        # normalize dividing by a constant
        return ConstantNormalization(config)



class NormalizationMethod(ABC):
    '''
    Abstract class for image normalization
    '''

    def __init__(self):
        '''
        Constructor
        '''
        super(NormalizationMethod, self).__init__()

    def normalize(self, image):
        '''
        Implement this method to normalize an image
        '''
        return np.asarray(image, dtype=np.float32)


class Z_Normalization(NormalizationMethod):
    '''
    Normalize to zero mean and unit variance
    '''

    def __init__(self):
        '''
        Constructor
        '''
        super(Z_Normalization, self).__init__()

    def normalize(self, image):
        '''
        Normalize to zero mean and unit variance
        '''
        image = super().normalize(image)
        # get mean and std
        mean_ = np.mean(image.flatten())
        std_ = np.std(image.flatten()) + 0.000001

        return (image - mean_) / std_


class MaximumNormalization(NormalizationMethod):
    '''
    Normalize with the maximum value
    '''

    def __init__(self):
        '''
        Constructor
        '''
        super(MaximumNormalization, self).__init__()


    def normalize(self, image):
        '''
        Normalize with the maximum value
        '''
        image = super().normalize(image)
        # get maximum value
        max_ = np.max(image.flatten()) + 0.000001

        return image / max_


class RangeNormalization(NormalizationMethod):
    '''
    Normalize with the minimum and maximum values
    '''

    def __init__(self):
        '''
        Constructor
        '''
        super(RangeNormalization, self).__init__()
        

    def normalize(self, image):
        '''
        Normalize with the minimum and maximum values
        '''
        image = super().normalize(image)
        # get min and max values
        min_ = np.min(image.flatten())
        max_ = np.max(image.flatten())

        return (image - min_) / (max_ - min_ + 0.000001)


class ConstantNormalization(NormalizationMethod):
    '''
    Normalize with the minimum and maximum values
    '''

    def __init__(self, config):
        '''
        Constructor
        '''
        super(ConstantNormalization, self).__init__()

        # constant for normalization
        if 'normalization-constant' in config['training']:
            self.constant = float(config['training']['normalization-constant'])
        else:
            self.constant = 255.0
        

    def normalize(self, image):
        '''
        Normalize dividing by 255
        '''
        image = super().normalize(image)
        return image / self.constant