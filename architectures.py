
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from skimage import filters
from os import path

from file_access import parse_boolean, get_list_of_strings_from_string, makedir_if_doesnt_exist
from util import bayesian_eval_mode, Flatten, get_final_activation_unit, get_activation_unit, get_pooling_layer, get_dropout, get_normalization_strategy

from ast import literal_eval as make_tuple




class SegmentationNetwork(nn.Module):
    '''
    Abstract class defining some key methods for segmenting B-scans.
    '''

    def __init__(self, config):
        '''
        Constructor of the base class.
        '''
        super(SegmentationNetwork, self).__init__()

        # set the name of the model
        self.name = 'model'
        # set the target
        self.target = config['experiment']['target']

        # setup default configuration
        self.n_classes = 2              # number of output classes
        self.is_deconv = False          # deconvolutions or upsampling
        self.is_batchnorm = True        # batch normalization
        self.dropout = 0.0              # dropout probability
        self.use_otsu = False           # a boolean indicating if we need to use Otsu or not
        self.final_activation = None    # final activation
        self.ignore_class = -1          # by default, no class is ignored

        # normalization strategy
        self.normalization_method = get_normalization_strategy(config)

        # change configuration if available in the file
        if 'n-classes' in config['architecture']:
            self.n_classes = int(config['architecture']['n-classes'])
        if 'use-deconvolution' in config['architecture']:
            self.is_deconv = parse_boolean(config['architecture']['use-deconvolution'])
        if 'batch-norm' in config['architecture']:
            self.is_batchnorm = parse_boolean(config['architecture']['batch-norm'])
        if 'dropout' in config['training']:
            self.dropout = float(config['training']['dropout'])
        if 'use-otsu' in config['experiment']:
            self.use_otsu = parse_boolean(config['experiment']['use-otsu'])
        if 'final-activation' in config['architecture']:
            self.final_activation = self.read_activation_from_config_file(config)
        if 'ignore-class' in config['training']:
            self.ignore_class = int(config['training']['ignore-class'])


    def read_activation_from_config_file(self, config):
        '''
        Read the final activation from a config file
        '''

        return get_final_activation_unit(config['architecture']['final-activation'])


#################################################################################################################
#################################################################################################################


class UnetConvBlock(nn.Module):
    '''
    Convolutional block of a U-Net:
    Conv2d - Batch normalization (optional) - ReLU
    Conv2D - Batch normalization (optional) - ReLU
    Basic Dropout (optional)
    '''

    def __init__(self, in_size, out_size, is_batchnorm, dropout, activation='relu'):
        '''
        Constructor of the convolutional block
        '''
        super(UnetConvBlock, self).__init__()

        # Convolutional layer with IN_SIZE --> OUT_SIZE
        conv1 = nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=3, stride=1, padding=1, bias=not(is_batchnorm))
        # Activation unit
        activ_unit1 = get_activation_unit(activation)
        # Add batch normalization if necessary
        if is_batchnorm:
            self.conv1 = nn.Sequential(conv1, nn.BatchNorm2d(out_size), activ_unit1)
        else:
            self.conv1 = nn.Sequential(conv1, activ_unit1)

        # Convolutional layer with OUT_SIZE --> OUT_SIZE
        conv2 = nn.Conv2d(in_channels=out_size, out_channels=out_size, kernel_size=3, stride=1, padding=1, bias=not(is_batchnorm))
        # Activation unit
        activ_unit2 = get_activation_unit(activation)
        # Add batch normalization if necessary
        if is_batchnorm:
            self.conv2 = nn.Sequential(conv2, nn.BatchNorm2d(out_size), activ_unit2)
        else:
            self.conv2 = nn.Sequential(conv2, activ_unit2)

        # Dropout
        if dropout > 0.0:
            self.drop = nn.Dropout(dropout)
        else:
            self.drop = None


    def forward(self, inputs):
        '''
        Do a forward pass
        '''
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        if not (self.drop is None):
            outputs = self.drop(outputs)
        return outputs



class UnetUpsampling(nn.Module):
    '''
    Upsampling block of a U-Net:
    TransposeConvolution / Upsampling
    Convolutional block
    '''

    def __init__(self, in_size, out_size, upsample_size, is_deconv, dropout, is_batchnorm, activation='relu', upsampling_type='nearest'):
        '''
        Constructor of the upsampling block
        '''
        super(UnetUpsampling, self).__init__()

        if is_deconv:
            # first a transposed convolution
            self.up = nn.ConvTranspose2d(upsample_size, upsample_size, kernel_size=2, stride=2)
            # and then a convolution
            conv = nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=3, stride=1, padding=1, bias=not(is_batchnorm))   # convolution
            activ_unit1 = get_activation_unit(activation)                                                                              # activation
            if is_batchnorm:                                                                                                           # batch norm
                self.conv = nn.Sequential(conv, nn.BatchNorm2d(out_size), activ_unit1)
            else:
                self.conv = nn.Sequential(conv, activ_unit1)
        else:
            # first an upsampling operation
            self.up = None
        # and then a convolutional block
        self.conv = UnetConvBlock(in_size, out_size, is_batchnorm, dropout, activation)


    def forward(self, from_skip_connection, from_lower_size):
        '''
        Do a forward pass
        '''

        # upsampling the input from the previous layer
        if self.up is None:
            rescaled_input = F.interpolate(from_lower_size, scale_factor=2)
        else:
            rescaled_input = self.up(from_lower_size)
        # verify the differences between the two tensors and apply padding
        offset = rescaled_input.size()[2] - from_skip_connection.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        from_skip_connection = F.pad(from_skip_connection, padding)
        # concatenate and apply the convolutional block on it
        return self.conv(torch.cat([from_skip_connection, rescaled_input], 1))




#################################################################################################################
#################################################################################################################



class Brunet(SegmentationNetwork):
    '''
    BRU-Net architecture. 
    Based on https://arxiv.org/pdf/1707.04931.pdf
    '''

    def __init__(self, config):
        '''
        Constructor of the BRU-Net architecture
        '''

        # default configuration of a BRU-Net        
        config['architecture']['batch-norm'] = 'True'
        config['architecture']['use-deconvolution'] = 'False'
        
        super(Brunet, self).__init__(config)

        # set the default configuration
        filters=[32, 64, 128, 256, 512, 512]    # number of channels of each conv layer
        in_channels=1                           # number of input channels

        # change configuration if available in the file
        if 'filters' in config['architecture']:
            filters = np.fromstring( config['architecture']['filters'], dtype=int, sep=',' )
            filters = filters.tolist()

        # Activation function to produce the scores
        self.final_activation = get_final_activation_unit(config['architecture']['final-activation'])

        # Input block
        self.input_block = BrunetInputBlock(in_size=in_channels, out_size=filters[0], is_batchnorm=self.is_batchnorm, dropout=0.0)
        # Encoder branch, with the block-d elements
        self.block_d_1 = BrunetBlockD(in_size=filters[0], out_size=(filters[1]-in_channels), is_batchnorm=self.is_batchnorm, dropout=0.0)
        self.cat_1 = BrunetCat(2)
        self.block_d_2 = BrunetBlockD(in_size=filters[1], out_size=(filters[2]-in_channels), is_batchnorm=self.is_batchnorm, dropout=0.0)
        self.cat_2 = BrunetCat(4)
        self.block_d_3 = BrunetBlockD(in_size=filters[2], out_size=(filters[3]-in_channels), is_batchnorm=self.is_batchnorm, dropout=0.0)
        self.cat_3 = BrunetCat(8)
        self.block_d_4 = BrunetBlockD(in_size=filters[3], out_size=(filters[4]-in_channels), is_batchnorm=self.is_batchnorm, dropout=0.0)
        self.cat_4 = BrunetCat(16)
        self.block_d_5 = BrunetBlockD(in_size=filters[4], out_size=(filters[5]-in_channels), is_batchnorm=self.is_batchnorm, dropout=0.0)
        self.cat_5 = BrunetCat(32)
        # Bottleneck layer
        self.block_u_5 = BrunetBlockU(in_size=filters[5], out_size=filters[5], is_batchnorm=self.is_batchnorm, dropout=self.dropout)
        # Decoder branch, with the block-u elements    
        self.block_u_4 = BrunetBlockUandCat(in_size=filters[4] + filters[5], out_size=filters[4], is_batchnorm=self.is_batchnorm, dropout=0.0)
        self.block_u_3 = BrunetBlockUandCat(in_size=filters[3] + filters[4], out_size=filters[3], is_batchnorm=self.is_batchnorm, dropout=0.0)
        self.block_u_2 = BrunetBlockUandCat(in_size=filters[2] + filters[3], out_size=filters[2], is_batchnorm=self.is_batchnorm, dropout=0.0)
        self.block_u_1 = BrunetBlockUandCat(in_size=filters[1] + filters[2], out_size=filters[1], is_batchnorm=self.is_batchnorm, dropout=0.0)
        # Output block
        self.output_block = BrunetOutputBlock(in_size=filters[1], out_size=self.n_classes, is_batchnorm=self.is_batchnorm, dropout=0.0)


    def forward(self, images):
        '''
        Do a forward pass
        '''
        # Encoder branch (including the input block and all the block-d elements)
        outs_1 = self.cat_1(images, self.block_d_1(self.input_block(images)))
        outs_2 = self.cat_2(images, self.block_d_2(outs_1))
        outs_3 = self.cat_3(images, self.block_d_3(outs_2))
        outs_4 = self.cat_4(images, self.block_d_4(outs_3))
        outs_5 = self.cat_5(images, self.block_d_5(outs_4))
        # Bottleneck layer
        up_5 = self.block_u_5(outs_5)
        # Decoder branch
        up_4 = self.block_u_4(outs_4, up_5)
        up_3 = self.block_u_3(outs_3, up_4)
        up_2 = self.block_u_2(outs_2, up_3)
        up_1 = self.output_block(self.block_u_1(outs_1, up_2))
        # Activation function
        up1 = self.final_activation(up1)
        return up_1



class BrunetInputBlock(nn.Module):
    '''
    This class models the input block from the BRU-Net
    (see https://arxiv.org/pdf/1707.04931.pdf)
    '''

    def __init__(self, in_size, out_size=32, is_batchnorm=True, dropout=0.0, activation='relu'):
        '''
        Constructor of the Input Block of the U-Net
        '''
        super(BrunetInputBlock, self).__init__()

        # Convolutional layer
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=5, stride=1, padding=2)
        # Batch normalization (if necessary)
        self.bn = None
        if is_batchnorm:
            self.bn = nn.BatchNorm2d(out_size)           
        # Activation unit
        self.activation_unit = get_activation_unit(activation)
        # Dropout (if necessary)
        self.drop = None
        if dropout > 0.0:
            self.drop = nn.Dropout(dropout)    


    def forward(self, inputs):
        '''
        Do a forward pass
        '''

        # convolutional layer
        outs = self.conv1(inputs)
        # batch normalization
        if not (self.bn is None):
            outs = self.bn(outs)
        # activation unit
        outs = self.activation_unit(outs)
        # dropout
        if not (self.drop is None):
            outs = self.drop(outs)
        return outs



class BrunetOutputBlock(nn.Module):
    '''
    This class models the output block of the BRU-Net
    (see https://arxiv.org/pdf/1707.04931.pdf)
    '''

    def __init__(self, in_size, out_size=2, is_batchnorm=True, dropout=0.0, activation='relu'):
        '''
        Constructor of the BRU-Net output block
        '''
        super(BrunetOutputBlock, self).__init__()

        # convolutional layer
        self.conv1 = nn.Conv2d(in_size, in_size, kernel_size=3, stride=1, padding=1, bias=not(is_batchnorm))
        # batch normalization (optional)
        self.bn = None
        if is_batchnorm:
            self.bn = nn.BatchNorm2d(in_size)
        # activation unit
        self.activation_unit = get_activation_unit(activation)
        # dropout (optional)
        self.drop = None
        if dropout > 0.0:
            self.drop = nn.Dropout(dropout) 
        # a second convolutional layer
        self.conv2 = nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=not(is_batchnorm))


    def forward(self, inputs):
        '''
        Do a forward pass
        '''

        # convolutional layer
        outs = self.conv1(inputs)
        # batch normalization
        if not (self.bn is None):
            outs = self.bn(outs)
        # activation unit
        outs = self.activation_unit(outs)
        # dropout
        if not (self.drop is None):
            outs = self.drop(outs)
        # convolutional layer
        outs = self.conv2(outs)
        return outs



class BrunetBlock(nn.Module):
    '''
    Common BRU-Net block for both the Block-U and the Block-D
    '''

    def __init__(self, in_size, out_size, is_batchnorm=True, dropout=0.0, activation='relu'):
        '''
        Constructor of the common BRU-Net block
        '''
        super(BrunetBlock, self).__init__()

        # first convolutional block
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0)
        # three dilated convolutions with different dilation factors
        self.dil_conv1 = nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=1, dilation=1, bias=not(is_batchnorm))
        self.dil_conv3 = nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=3, dilation=3, bias=not(is_batchnorm))
        self.dil_conv5 = nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=5, dilation=5, bias=not(is_batchnorm))
        # optional batch normalization
        self.bn = None
        if is_batchnorm:
            self.bn = nn.BatchNorm2d(out_size)
        # activation unit
        self.activation_unit = get_activation_unit(activation)


    def forward(self, inputs):
        '''
        Do a forward pass
        '''

        # first convolutional block
        outs = self.conv1(inputs)
        # three dilated convolutions
        outs_1 = self.dil_conv1(outs)
        outs_3 = self.dil_conv3(outs)
        outs_5 = self.dil_conv5(outs)
        # sum of the dilated convolutions and the first convolution
        outs = outs + outs_1 + outs_3 + outs_5
        # batch normalization
        if not (self.bn is None):
            outs = self.bn(outs)
        # activation
        outs = self.activation_unit(outs)

        return outs



class BrunetBlockD(nn.Module):
    '''
    Block-D for the BRU-Net
    (see https://arxiv.org/pdf/1707.04931.pdf)
    '''

    def __init__(self, in_size, out_size, is_batchnorm=True, dropout=0.0, activation='relu'):
        '''
        Constructor of the Block-D
        '''
        super(BrunetBlockD, self).__init__()

        # first create a common BrunetBlock
        self.block = BrunetBlock(in_size, out_size, is_batchnorm, dropout, activation)
        # optional dropout
        self.drop = None
        if dropout > 0.0:
            self.drop = nn.Dropout(dropout) 
        # convolution
        self.conv = nn.Conv2d(out_size, out_size, kernel_size=1, stride=1, padding=0, bias=not(is_batchnorm))
        # max pooling
        self.max_pool = nn.MaxPool2d(kernel_size=2) # or 3?
    

    def forward(self, inputs):
        '''
        Do a forward pass
        '''

        # run the common block
        outs = self.block(inputs)
        # dropout
        if not (self.drop is None):
            outs = self.drop(outs)
        # last convolution
        outs = self.conv(outs)
        # max pooling
        return self.max_pool(outs)



class BrunetBlockU(nn.Module):
    '''
    Block-U for the BRU-Net
    (see https://arxiv.org/pdf/1707.04931.pdf)
    '''

    def __init__(self, in_size, out_size, is_batchnorm=True, dropout=0.0, activation='relu', upsampling_type='nearest'):
        '''
        Constructor of the block D
        '''
        super(BrunetBlockU, self).__init__()

        # first create a common BrunetBlock
        self.block = BrunetBlock(in_size, out_size, is_batchnorm, dropout, activation)
        # dropout (optional)
        self.drop = None
        if dropout > 0.0:
            self.drop = nn.Dropout(dropout) 
        # convolution
        self.conv = nn.Conv2d(out_size, out_size, kernel_size=1, stride=1, padding=0, bias=not(is_batchnorm))


    def forward(self, inputs):
        '''
        Do a forward pass
        '''

        # common BrunetBlock
        outs = self.block(inputs)
        # dropout
        if not (self.drop is None):
            outs = self.drop(outs)
        # convolution
        outs = self.conv(outs)
        # upsampling
        return F.interpolate(outs, scale_factor=2)



class BrunetBlockUandCat(nn.Module):
    '''
    A Block-U with previous concatenation from a skip connection
    '''

    def __init__(self, in_size, out_size, is_batchnorm=True, dropout=0.0, activation='relu', upsampling_type='nearest'):
        '''
        Construction of the Block-U + concatenation block
        '''
        super(BrunetBlockUandCat, self).__init__()

        # we only need the Block-U
        self.block = BrunetBlockU(in_size, out_size, is_batchnorm, dropout)


    def forward(self, from_skip_connection, from_bottom):
        '''
        Do a forward pass
        '''

        # pad the input if necessary
        offset = from_skip_connection.size()[2] - from_bottom.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        padding[2] = offset
        from_bottom = F.pad(from_bottom, padding)

        # concatenate them
        inputs = torch.cat([from_skip_connection, from_bottom], 1)

        # use this as input for the block-U
        return self.block(inputs)



class BrunetCat(nn.Module):
    '''
    This block concatenates a resized image with the output
    of a Block-D layer
    '''

    def __init__(self, downsampling):
        '''
        Constructor of the BRU-Net concatenation
        '''
        super(BrunetCat, self).__init__()

        # resize the image according to the downsampling factor
        self.img_resize = nn.AvgPool2d(downsampling)


    def forward(self, images, from_top):
        '''
        Do a forward pass
        '''

        # resize the image
        images = self.img_resize(images)
        # concatenate
        return torch.cat([images, from_top], 1)



#################################################################################################################
#################################################################################################################



class CustomizedUnet(SegmentationNetwork):
    '''
    CustomizedUnet U-Net architecture
    Same as in the original paper (https://arxiv.org/pdf/1505.04597.pdf)
    but you can modify:
    -- activation functions
    -- pooling layers
    -- dropout
    '''

    def __init__(self, config):
        '''
        Constructor of the standard U-Net
        '''

        super(CustomizedUnet, self).__init__(config)

        # set the default configuration
        self.dropout = [0.0, 0.0, 0.0, 0.0, self.dropout, 0.0, 0.0, 0.0, 0.0]   # dropout rate of each layer
        filters_encoder = [64, 128, 256, 512, 1024]                             # number of channels of each conv block in the encoder
        filters_decoder = [64, 128, 256, 512]                                   # number of channels of each conv block in the decoder
        activation = 'relu'                                                     # activation units
        pooling = 'max'                                                         # type of pooling

        # change configuration if available in the file
        if 'filters-encoder' in config['architecture']:
            filters_encoder = np.fromstring( config['architecture']['filters-encoder'], dtype=int, sep=',' )
        if 'filters-decoder' in config['architecture']:
            filters_decoder = np.fromstring( config['architecture']['filters-decoder'], dtype=int, sep=',' )
        if 'dropout-list' in config['architecture']:
            self.dropout = np.fromstring( config['architecture']['dropout-list'], dtype=float, sep=',' )
        if 'activation' in config['architecture']:
            activation = config['architecture']['activation']
        if 'pooling' in config['architecture']:
            pooling = config['architecture']['pooling']

        # Activation function to produce the scores
        self.final_activation = get_final_activation_unit(config['architecture']['final-activation'])

        # downsampling
        self.conv1 = UnetConvBlock(1, int(filters_encoder[0]), self.is_batchnorm, self.dropout[0], activation)
        self.pool1 = get_pooling_layer(pooling_name=pooling, kernel_size=2)
        self.conv2 = UnetConvBlock(int(filters_encoder[0]), int(filters_encoder[1]), self.is_batchnorm, self.dropout[1], activation)
        self.pool2 = get_pooling_layer(pooling_name=pooling, kernel_size=2)
        self.conv3 = UnetConvBlock(int(filters_encoder[1]), int(filters_encoder[2]), self.is_batchnorm, self.dropout[2], activation)
        self.pool3 = get_pooling_layer(pooling_name=pooling, kernel_size=2)
        self.conv4 = UnetConvBlock(int(filters_encoder[2]), int(filters_encoder[3]), self.is_batchnorm, self.dropout[3], activation)
        self.pool4 = get_pooling_layer(pooling_name=pooling, kernel_size=2)
        # intermediate module
        self.conv5 = UnetConvBlock(int(filters_encoder[3]), int(filters_encoder[4]), self.is_batchnorm, self.dropout[4], activation)
        # upsampling
        self.up_concat4 = UnetUpsampling(int(filters_encoder[4]) + int(filters_encoder[3]), int(filters_decoder[3]), int(filters_encoder[4]), self.is_deconv, self.dropout[5], self.is_batchnorm, activation)
        self.up_concat3 = UnetUpsampling(int(filters_decoder[3]) + int(filters_encoder[2]), int(filters_decoder[2]), int(filters_encoder[3]), self.is_deconv, self.dropout[6], self.is_batchnorm, activation)
        self.up_concat2 = UnetUpsampling(int(filters_decoder[2]) + int(filters_encoder[1]), int(filters_decoder[1]), int(filters_encoder[2]), self.is_deconv, self.dropout[7], self.is_batchnorm, activation)
        self.up_concat1 = UnetUpsampling(int(filters_decoder[1]) + int(filters_encoder[0]), int(filters_decoder[0]), int(filters_encoder[1]), self.is_deconv, self.dropout[8], self.is_batchnorm, activation)
        # final conv (without any concat)
        self.final = nn.Conv2d(int(filters_decoder[0]), self.n_classes, 1)


    def forward(self, inputs):
        '''
        Do a forward pass
        '''

        # downsampling
        conv1 = self.conv1(inputs)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool2(conv3)
        conv4 = self.conv4(pool3)
        pool4 = self.pool2(conv4)
        # intermediate module with dropout
        conv5 = self.conv5(pool4)
        # upsampling
        up4 = self.up_concat4(conv4, conv5)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)
        # get the segmentation
        up1 = self.final(up1)
        # apply the activation
        up1 = self.final_activation(up1)

        return up1


#################################################################################################################
#################################################################################################################


class StandardUnet(CustomizedUnet):
    '''
    Standard U-Net architecture
    Same as in the original paper (https://arxiv.org/pdf/1505.04597.pdf)
    but with batch normalization (otherwise, it fails)
    '''

    def __init__(self, config):
        '''
        Constructor of the standard U-Net
        '''

        # change configuration to fit the standard U-Net
        config['architecture']['filters'] = '64, 128, 256, 512, 1024'
        config['architecture']['use-deconvolution'] = 'True'
        config['architecture']['batch-norm'] = 'True'
        config['training']['dropout'] = '0.0'
        config['architecture']['activation'] = 'relu'
        config['architecture']['pooling'] = 'max'

        super(StandardUnet, self).__init__(config)


#################################################################################################################
#################################################################################################################


class U2net(CustomizedUnet):
    '''
    U2-Net architecture
    Same as in the paper (https://arxiv.org/pdf/1901.07929.pdf)
    '''

    def __init__(self, config):
        '''
        Constructor of the standard U-Net
        '''

        # change configuration to fit the U2-Net
        config['architecture']['filters'] = '64, 128, 256, 512, 1024'
        config['architecture']['use-deconvolution'] = 'False'
        config['architecture']['batch-norm'] = 'True'
        config['architecture']['dropout-list'] = '0.0, 0.1, 0.1, 0.1, 0.5, 0.1, 0.1, 0.1, 0.0'
        config['architecture']['activation'] = 'lrelu'
        config['architecture']['pooling'] = 'max'

        super(U2net, self).__init__(config)

