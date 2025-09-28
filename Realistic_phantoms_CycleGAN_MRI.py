# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 12:18:37 2024

@author: fanciicam
"""

'''Realistic phantoms MRI CycleGAN'''

#____________________________LIBRARIES_AND_DIRECTORIES_________________________

import SimpleITK as sitk
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from random import randint 
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import time
import h5py
import cv2
from image_similarity_measures import quality_metrics
import glob

folder_hdf5 = '//NAS-CARTCAS/camagnif/tesi/hdf5/'
folder_weights = '//NAS-CARTCAS/camagnif/tesi/Data/mri/weights/'
folder_results_train = '//NAS-CARTCAS/camagnif/tesi/Data/mri/results/train/'
folder_results_test = '//NAS-CARTCAS/camagnif/tesi/Data/mri/results/test/'

RESIZE_DIM = 256

#%% 

#__________________________________FUNCTIONS___________________________________

def load(fileName):
    reader = sitk.ImageFileReader() # reads an image file and returns an SItkimage
    reader.SetImageIO("MetaImageIO")
    reader.SetFileName(fileName)
    imageSitk = reader.Execute()
    imageNp = sitk.GetArrayViewFromImage(imageSitk) #320 x 260 x [slices]
    imageNp = np.moveaxis(imageNp, 0, 2) #260 x 320 x [slices]
    tensorImg = tf.convert_to_tensor(imageNp, dtype=tf.float32)

    return tensorImg

def resize(input_image):
    input_image = tf.image.resize(input_image, [RESIZE_DIM, RESIZE_DIM], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, preserve_aspect_ratio=False)
    return input_image

def resize_new(input_image):                    # CHANGE
    input_image = tf.image.resize(input_image, [220, 220], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, preserve_aspect_ratio=False)
    return input_image

# write on disk the mha volume from a tensorflow tensor
def writeSynthetic(imageTensor, synthFileName, spacing_z, write=False):
    #imageTensor = Invresize(imageTensor)
    imageNp = imageTensor.numpy()
    # write .mha
    writer = sitk.ImageFileWriter()
    writer.SetFileName(synthFileName)
    imageNp = np.moveaxis(imageNp, 2, 0)
    imageSitk = sitk.GetImageFromArray(imageNp)
    origin = [0.0, 0.0, 0.0]
    imageSitk.SetOrigin(origin)
    spacing = [1.0625, 1.0625, spacing_z]
    imageSitk.SetSpacing(spacing)
    imageSitk.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    if write:
        writer.Execute(imageSitk)
    return None

# from [-1;+1] back to original values range(0-100 for VIBE)
def merge_HU(predTensor):  #this name comes from CycleGAN CT...
    ct = predTensor[:, :]
    ctnew = 0.5*(100-(0))*(ct+1) + (0)  
    return ctnew


def generate_images(model, input_img):
    prediction = model(input_img, training=False)
    return prediction


# construction of the tensorflow tensor of the predicted volume + write of the mha volume 
def volume_build_patches(file, slices, modelXtoY, modelYtoX, synthFileName, mod, spacing_z, write=True):
    count = 0

    for i in slices:
        slice_in = file.get(mod)[i] # (348,348)
        true_slice = tf.convert_to_tensor(slice_in, dtype=tf.float32) # [348, 348]

        if mod=='COMBAT':
            recon_slice, tensor = reconstruct_img_patches(true_slice, modelXtoY)
        elif mod=='MRI':
            recon_slice, tensor = reconstruct_img_patches(true_slice, modelYtoX)
        prediction = merge_HU(recon_slice)
        
        if count == 0:            
            imageTensorPred = tf.expand_dims(prediction, axis=2) # If this is the first slice of a new volume, create the tensor
        elif count < len(slices)-1:           
            imageTensorPred = tf.concat([imageTensorPred, tf.expand_dims(prediction, 2)], 2)  # If this is a middle slice of the current volume, stack into the tensor
        elif count == len(slices)-1:            
            imageTensorPred = tf.concat([imageTensorPred, tf.expand_dims(prediction, 2)], 2) # if this is the last slice of the current volume, stack and create .mha file + metrics
        
            writeSynthetic(imageTensorPred, synthFileName,  spacing_z, write=write)           
            
            count = -1  # new volume
        count = count + 1
        
    return imageTensorPred


# construction of the tensorflow tensor of the predicted volume + write of the mha volume 
def volume_build_full(file, slices, modelXtoY, modelYtoX, synthFileName, mod, spacing_z, write=True):
    count = 0
        
    for i in slices:
        slice_in = file.get(mod)[i] # (348,348)
        true_slice = tf.convert_to_tensor(slice_in, dtype=tf.float32) # [348, 348]

        if mod=='COMBAT':
            recon_slice = reconstruct_img_full(true_slice, modelXtoY, 'mha')
        elif mod=='MRI':
            recon_slice = reconstruct_img_full(true_slice, modelYtoX, 'mha')
        prediction = merge_HU(recon_slice)
        
        if count == 0:            
            imageTensorPred = tf.expand_dims(prediction, axis=2) # If this is the first slice of a new volume, create the tensor
        elif count < len(slices)-1:           
            imageTensorPred = tf.concat([imageTensorPred, tf.expand_dims(prediction, 2)], 2)  # If this is a middle slice of the current volume, stack into the tensor
        elif count == len(slices)-1:            
            imageTensorPred = tf.concat([imageTensorPred, tf.expand_dims(prediction, 2)], 2) # if this is the last slice of the current volume, stack and create .mha file + metrics

            writeSynthetic(imageTensorPred, synthFileName,  spacing_z, write=write)
            
            count = -1  # new volume
        count = count + 1
        
    return imageTensorPred

# extract from volume the slice i 
def extract_slice(volume, i):
    slice_out = tf.slice(volume, [0, 0, i], [volume.shape[0], volume.shape[1], 1])
    return slice_out


# add a dimension in position "axis" to the input img as tensorflow tensor
def expand_dim(input_img, axis):
    output = tf.expand_dims(input_img, axis) 
    return output

# Add padding
def add_padding(image, pad_size):
    return tf.pad(image, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], mode='CONSTANT', constant_values=-1)

# Remove padding
def remove_padding(image, pad_size):
    return image[:, pad_size:-pad_size, pad_size:-pad_size, :]

def normalize(input_image):
    input_image = 2*((input_image-np.min(input_image)) / (np.max(input_image)-np.min(input_image))) - 1
    return input_image

#%% 

@tf.function
def extract_patches_inverse(shape, patches, stride, size):
    _x = tf.zeros(shape)
    _y = tf.image.extract_patches(images=_x,
                                sizes=[1, size, size, 1],
                                strides=[1, stride, stride, 1],
                                rates=[1, 1, 1, 1],
                                padding = 'VALID')
    grad = tf.gradients(_y, _x)[0]
    # Avoid division by 0
    grad = tf.where(tf.abs(grad) < 1e-7, tf.ones_like(grad), grad)
    # Avoid Nan in output
    output = tf.gradients(_y, _x, grad_ys=patches)[0] / grad
    output = tf.where(tf.math.is_nan(output), tf.zeros_like(output), output)
    return output

#%%

def reconstruct_img_patches(slice_in, model):
    # slice_in arrives from hdf5 [348, 348] for XCAT as a tensor
    size = 220
    stride = 32
    #pad_size = (size - stride) // 2  # padding dimension
    
    slice_in = tf.stack([slice_in], 0) # [1, 348, 348]
    slice_in = tf.stack([slice_in], 3) # [1, 348, 348, 1]
    #padded_slice_in = add_padding(slice_in, pad_size)
    
    patches = tf.image.extract_patches(images = slice_in,
                               sizes=[1, size, size, 1],
                               strides=[1, stride, stride, 1],
                               rates=[1, 1, 1, 1],
                               padding='VALID')
    n_patches = patches.shape[1]*patches.shape[2]
    patch = tf.reshape(patches, shape=(n_patches, size, size, 1)) #9
    i = 0
    patch_temp = expand_dim(patch[i], 0) #patch[i]: [220, 220, 1] patch_temp: [1, 220, 220, 1]
    tensor = expand_dim(tf.squeeze(resize_new(model(resize(patch_temp)))),0)
    for i in range(1,n_patches): 
        patch_temp = expand_dim(patch[i], 0)
        tensor = tf.concat([tensor, expand_dim(tf.squeeze(resize_new(model(resize(patch_temp)))),0)],0) 
        
        # patch_temp is [1,220,220,1], resize and model are [1,256,256,1], resize_new is [1,220,220,1]
        # squeeze is [220, 220], expand is [1, 220, 220] -> tensor is [npatches, 220, 220, 1]
    
    patch_new = tf.reshape(tensor, shape = (1, patches.shape[1], patches.shape[2], size*size)) # (1,3,3,65536)
    images_reconstructed = extract_patches_inverse((1, slice_in.shape[1], slice_in.shape[2], 1), patch_new, stride, size)
    #images_reconstructed = remove_padding(images_reconstructed, pad_size)

    return tf.squeeze(images_reconstructed), tensor # [348, 348]

def reconstruct_img_full(slice_in, model, mod2):
    
    dim = slice_in.shape[0]
    dim2 = slice_in.shape[1]

    slice_in = tf.stack([slice_in], 0) # [1, 256, 256]
    slice_in = tf.stack([slice_in], 3) # [1, 256, 256, 1]
    slice_in = resize(slice_in) # (256, 256)
    if mod2 == 'mha':
        slice_in = normalize(slice_in)
    prediction = model(slice_in) # [256, 256]
    prediction = invresize(prediction, dim, dim2)
    prediction = tf.squeeze(prediction)
    
    return prediction # [256, 256]

def invresize(input_image, dim, dim2):
    input_image = tf.image.resize(input_image, [dim, dim2], method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=False)
    return input_image

def get_transpose(slice_in):
    slice_t = tf.transpose(slice_in, perm=[2, 0, 1])
    return slice_t

# SSIM
def ssim(image_pred, image_gt):
    return quality_metrics.ssim(image_pred, image_gt, 100) # 100 is the range of values of the input img
                                                           # 100 if the img is not normalized, 2 if the img is normalized [-1;+1]
# FSIM
def fsim(image_pred, image_gt):
    fsim = quality_metrics.fsim(image_pred, image_gt, 100) # same
    return fsim

def edge_ratios(img1, img2, threshold=0.4):

    # edge extraction
    sobel_filter = tf.constant([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=tf.float32)
    img1_edges = tf.nn.conv2d(np.expand_dims(img1, 0), tf.expand_dims(tf.expand_dims(sobel_filter, -1), -1), strides=[1, 1, 1, 1], padding='SAME')
    img2_edges = tf.nn.conv2d(np.expand_dims(img2, 0), tf.expand_dims(tf.expand_dims(sobel_filter, -1), -1), strides=[1, 1, 1, 1], padding='SAME')

    # edges are positive and negative, need abs since I will then apply a threshold for binary imgs
    edge_gt = np.abs(tf.squeeze(img1_edges))
    edge_pred = np.abs(tf.squeeze(img2_edges))
    _, binary_img_gt = cv2.threshold(edge_gt, threshold, 1, cv2.THRESH_BINARY)
    _, binary_img_pred = cv2.threshold(edge_pred, threshold, 1, cv2.THRESH_BINARY)
    inter = cv2.bitwise_and(binary_img_gt, binary_img_pred) # edge map in common between edges1 and edges2
    inter = np.abs(inter)
    _, binary_img_inter = cv2.threshold(inter, threshold * 0.625, 1, cv2.THRESH_BINARY)  # 0.625 è un valore approssimato per mantenere l'intersezione valida
    
    # compute number of pixels = 1 belonging to edges
    tot_gt = np.sum(binary_img_gt)
    tot_pred = np.sum(binary_img_pred)
    tot_inter = np.sum(binary_img_inter)

    # EPR and EGR
    epr = tot_inter / tot_gt
    egr = tot_pred / tot_gt

    return epr, egr


# mean absolute error where mask == 1
def mean_absolute_error(image_pred, image_gt, mask):
    diff = np.abs(image_pred - image_gt)
    masked_diff = np.where(mask == 1, diff, 0) # = diff where the corresponding element in mask is 1, otherwis 0.
    return np.mean(masked_diff)


# Compute all metrics paired 
def metrics_paired(image_pred, image_gt, mask):
    ssim_metric = ssim(image_pred, image_gt)
    fsim_metric = fsim(image_pred, image_gt)
    image_pred_edge = normalize(image_pred)
    image_gt_edge = normalize(image_gt)
    epr, egr = edge_ratios(image_pred_edge, image_gt_edge)
    mae = mean_absolute_error(image_pred, image_gt, mask)
    return ssim_metric, fsim_metric, epr, egr, mae

# Compute all metrics unpaired
def metrics_unpaired(volume_true, volume_pred, liver_mask):
    counts_true = np.histogram(volume_true.numpy(), bins=100)
    counts_pred = np.histogram(volume_pred.numpy(), bins=100)

    correlation = np.corrcoef(counts_true[0], counts_pred[0])
    HistCC = correlation[1,0]
    
    mean_liver = np.mean(volume_pred[liver_mask==1])
    std_liver = np.std(volume_pred[liver_mask==1])

    return HistCC, std_liver, mean_liver

#%%

#________________________PLOT_RESULTS__________________________________________
# this will be done at each epoch to save and track losses during training

def plot_loss(ax, data, labels, title):
    # Colors
    colors = ["orange", "purple"]
    for i, label in enumerate(labels):
        ax.plot(data["Epoch"], data[label], color=colors[i], label=label)
    ax.set_title(title)
    ax.legend()
    ax.set(xlabel='Epochs', ylabel='Loss')

def plot_losses(results_loss_train_ep, epoch):# Plot construction

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    plot_loss(axs[0, 0], results_loss_train_ep, ["D_X_loss", "D_Y_loss"], 'Discriminators')
    plot_loss(axs[0, 1], results_loss_train_ep, ["G_XtoY_gan_loss", "G_YtoX_gan_loss"], 'Adversarial')
    plot_loss(axs[0, 2], results_loss_train_ep, ["cycle_X_loss", "cycle_Y_loss"], 'Cycle')
    plot_loss(axs[1, 0], results_loss_train_ep, ["int_x_loss", "int_y_loss"], 'Identity')
    plot_loss(axs[1, 1], results_loss_train_ep, ["total_loss_G"], 'Total G')
    plot_loss(axs[1, 2], results_loss_train_ep, ["total_loss_F"], 'Total F')
    
    # Add legend
    fig.legend(["$D_X$ loss", "$D_Y$ loss", "$G_{XtoY}$ GAN loss", "$G_{YtoX}$ GAN loss",
                "$Cycle_X$ loss", "$Cycle_Y$ loss", "$Int_X$ loss", "$Int_Y$ loss",
                "Total G loss", "Total F loss"],
               loc='lower center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=5)
    
    # Save plot 
    plt.subplots_adjust(hspace=0.3, wspace=0.3, top=0.9, left=0.1, right=0.9, bottom=0.1)
    fig.savefig(folder_results_train + 'loss_curves_training_' + str(epoch) + '_.png', bbox_inches='tight')
    
    return None

def plot_metrics(dict_for, dict_back, epoch):
    
    fig, axs = plt.subplots(2, 4, figsize=(15, 10))
    metrics_labels = ['SSIM', 'FSIM', 'EPR', 'EGR', 'MAE', 'HistCC', 'Liver_mean', 'Liver_std']
    colors = ['lightblue', 'lightgreen']
    
    for i, metric_label in enumerate(metrics_labels):
        row = i // 4
        col = i % 4
        
        axs[row, col].plot(dict_for["Patches"]["Epoch"], dict_for["Patches"][metric_label], color=colors[0], label='Patches')
        axs[row, col].plot(dict_for["Full_img"]["Epoch"], dict_for["Full_img"][metric_label], color=colors[1], label='Full Image')
        axs[row, col].set_title(metric_label + '_forward')
        axs[row, col].set_xlabel('Epochs')
        axs[row, col].set_ylabel('Metric Value')
        axs[row, col].legend()
        
    plt.tight_layout()
    plt.savefig(folder_results_test + 'metrics_curves_forward' + '_' + str(epoch) + '.png')
    plt.close()
    
    fig, axs = plt.subplots(2, 4, figsize=(15, 10))
    
    for i, metric_label in enumerate(metrics_labels):
        row = i // 4
        col = i % 4
        
        axs[row, col].plot(dict_back["Patches"]["Epoch"], dict_back["Patches"][metric_label], color=colors[0], label='Patches')
        axs[row, col].plot(dict_back["Full_img"]["Epoch"], dict_back["Full_img"][metric_label], color=colors[1], label='Full Image')
        axs[row, col].set_title(metric_label + '_backward')
        axs[row, col].set_xlabel('Epochs')
        axs[row, col].set_ylabel('Metric Value')
        axs[row, col].legend()
        
    plt.tight_layout()
    plt.savefig(folder_results_test + 'metrics_curves_backward' + '_' + str(epoch) + '.png')
    plt.close()

    return None

#%% 

#____________________________IMPORT_DATASET_AND_CHECK_PATCHES__________________

desired = 'MRI' # or MRI
file_to_check = desired + '_dataset_CycleGAN_train.h5'

h5f = h5py.File(folder_hdf5 + file_to_check, 'r')
list(h5f.keys())
print('Train 2D', desired, 'slices:', len(h5f.get(desired)))

# Extract data from the HDF5 file
nslices_to_viz = 4
slices_list = []
#mask_slices_list = []

for i in range(nslices_to_viz):
    rand_slice = randint(0, len(h5f.get(desired))-1)
    new_slice = h5f.get(desired)[rand_slice]
    slices_list.append(new_slice)
    #new_mask_slice = h5f.get("mask"+desired)[rand_slice]
    #mask_slices_list.append(new_mask_slice)


plt.figure(figsize=(10, 10))
title = [desired]
for i in range(nslices_to_viz):
    plt.subplot(1, nslices_to_viz, i+1) 
    plt.title(title[0])
    plt.imshow(slices_list[i], cmap='gray')
    plt.axis('off')
plt.show()


#plt.figure(figsize=(10, 10))
#title = ['mask'+desired]
#for i in range(nslices_to_viz):
#    plt.subplot(1, nslices_to_viz, i+1) 
#    plt.title(title[0])
#    plt.imshow(mask_slices_list[i], cmap='gray')
#    plt.axis('off')
#plt.show()

#%% 

# _________________________________CYCLEGAN____________________________________

input_img_size = (256,256,1)

class ReflectionPadding2D(layers.Layer):

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]
        return tf.pad(input_tensor, padding_tensor, mode="REFLECT")

# Weights initializer for the layers
kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
# Gamma initializer for instance normalization
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)


def residual_block(x, activation, kernel_initializer=kernel_init, kernel_size=(3, 3), strides=(1, 1), padding="valid", gamma_initializer=gamma_init, use_bias=False):
    dim = x.shape[-1]
    input_tensor = x

    x = ReflectionPadding2D()(input_tensor)
    x = layers.Conv2D(dim, kernel_size, strides=strides, kernel_initializer=kernel_initializer, padding=padding, use_bias=use_bias)(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = activation(x)

    x = ReflectionPadding2D()(x)
    x = layers.Conv2D(dim, kernel_size, strides=strides, kernel_initializer=kernel_initializer, padding=padding, use_bias=use_bias)(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = layers.add([input_tensor, x])
    return x


def downsample(x, filters, activation, kernel_initializer=kernel_init, kernel_size=(3, 3), strides=(2, 2), padding="same", gamma_initializer=gamma_init, use_bias=False): 
    x = layers.Conv2D(filters, kernel_size, strides=strides, kernel_initializer=kernel_initializer, padding=padding, use_bias=use_bias)(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x


def upsample(x, filters, activation, kernel_size=(3, 3), strides=(1, 1), padding="valid", kernel_initializer=kernel_init, gamma_initializer=gamma_init, use_bias=False):
    #dim = x.shape[-1]
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
    x = ReflectionPadding2D()(x)
    x = layers.Conv2D(filters, kernel_size, strides=strides, kernel_initializer=kernel_initializer, padding=padding, use_bias=use_bias)(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = activation(x)
    return x


def get_resnet_generator(filters=64, num_downsampling_blocks=2, num_residual_blocks=9, num_upsample_blocks=2, gamma_initializer=gamma_init):
    img_input = layers.Input(shape=input_img_size)
    #mask = layers.Input(shape=input_img_size)
    x = ReflectionPadding2D(padding=(3, 3))(img_input)
    x = layers.Conv2D(filters, (7, 7), kernel_initializer=kernel_init, use_bias=False)(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = layers.Activation("relu")(x)

    # Downsampling
    for _ in range(num_downsampling_blocks):
        filters *= 2
        x = downsample(x, filters=filters, activation=layers.Activation("relu"))

    # Residual blocks
    for _ in range(num_residual_blocks):
        x = residual_block(x, activation=layers.Activation("relu"))

    # Upsampling
    for _ in range(num_upsample_blocks):
        filters //= 2
        x = upsample(x, filters=filters, activation=layers.Activation("relu"))

    # Final block
    x = ReflectionPadding2D(padding=(3, 3))(x)
    x = layers.Conv2D(1, (7, 7), padding="valid")(x)
    x = layers.Activation("tanh")(x)
    #x = Multiply()([x, mask])

    model = keras.models.Model(img_input, x)
    return model


def get_discriminator(filters=64, kernel_initializer=kernel_init, num_downsampling=3):
    img_input = layers.Input(shape=input_img_size)
    x = layers.Conv2D(filters,(4, 4), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer)(img_input)
    x = layers.LeakyReLU(0.2)(x)

    num_filters = filters
    for num_downsample_block in range(3):
        num_filters *= 2
        if num_downsample_block < 2:
            x = downsample(x, filters=num_filters, activation=layers.LeakyReLU(0.2), kernel_size=(4, 4), strides=(2, 2))
        else:
            x = downsample(x, filters=num_filters, activation=layers.LeakyReLU(0.2), kernel_size=(4, 4), strides=(1, 1))

    x = layers.Conv2D(1, (4, 4), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer)(x)

    model = keras.models.Model(inputs=img_input, outputs=x)
    return model

#%%
#___________________________________NEW_CYCLEGAN_______________________________

input_img_size = (256,256,1)

class ReflectionPadding2D(layers.Layer):

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]
        return tf.pad(input_tensor, padding_tensor, mode="REFLECT")

# Weights initializer for the layers
kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
# Gamma initializer for instance normalization
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

def residual_block(x, activation, kernel_initializer=kernel_init, kernel_size=(3, 3), strides=(1, 1), padding="valid", gamma_initializer=gamma_init, use_bias=False):
    dim = x.shape[-1]
    input_tensor = x

    x = ReflectionPadding2D()(input_tensor)
    x = layers.Conv2D(dim, kernel_size, strides=strides, kernel_initializer=kernel_initializer, padding=padding, use_bias=use_bias)(x)
    x = GroupNormalization(groups=-1,gamma_initializer=gamma_initializer)(x)
    x = activation(x)

    x = ReflectionPadding2D()(x)
    x = layers.Conv2D(dim, kernel_size, strides=strides, kernel_initializer=kernel_initializer, padding=padding, use_bias=use_bias)(x)
    x = GroupNormalization(groups=-1,gamma_initializer=gamma_initializer)(x)
    x = layers.add([input_tensor, x])
    return x


def downsample(x, filters, activation, kernel_initializer=kernel_init, kernel_size=(3, 3), strides=(2, 2), padding="same", gamma_initializer=gamma_init, use_bias=False): 
    x = layers.Conv2D(filters, kernel_size, strides=strides, kernel_initializer=kernel_initializer, padding=padding, use_bias=use_bias)(x)
    x = GroupNormalization(groups=-1,gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x


def upsample(x, filters, activation, kernel_size=(3, 3), strides=(1, 1), padding="valid", kernel_initializer=kernel_init, gamma_initializer=gamma_init, use_bias=False):
    #dim = x.shape[-1]
    #x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
    x = layers.Conv2DTranspose(filters, kernel_size=3, strides=2, padding='same')(x)
    x = ReflectionPadding2D()(x)
    x = layers.Conv2D(filters, kernel_size, strides=strides, kernel_initializer=kernel_initializer, padding=padding, use_bias=use_bias)(x)
    x = GroupNormalization(groups=-1,gamma_initializer=gamma_initializer)(x)
    x = activation(x)
    return x


def get_resnet_generator(filters=64, num_downsampling_blocks=1, num_residual_blocks=6, num_upsample_blocks=1, gamma_initializer=gamma_init):
    img_input = layers.Input(shape=input_img_size)
    #mask = layers.Input(shape=input_img_size)
    x = ReflectionPadding2D(padding=(3, 3))(img_input)
    x = layers.Conv2D(filters, (7, 7), kernel_initializer=kernel_init, use_bias=False)(x)
    x = GroupNormalization(groups=-1,gamma_initializer=gamma_initializer)(x)
    x = layers.Activation("relu")(x)

    # Downsampling
    for _ in range(num_downsampling_blocks):
        filters *= 2
        x = downsample(x, filters=filters, activation=layers.Activation("relu"))

    # Residual blocks
    for _ in range(num_residual_blocks):
        x = residual_block(x, activation=layers.Activation("relu"))

    # Upsampling
    for _ in range(num_upsample_blocks):
        filters //= 2
        x = upsample(x, filters=filters, activation=layers.Activation("relu"))

    # Final block
    x = ReflectionPadding2D(padding=(3, 3))(x)
    x = layers.Conv2D(1, (7, 7), padding="valid")(x)
    x = layers.Activation("tanh")(x)
    #x = Multiply()([x, mask])

    model = keras.models.Model(img_input, x)
    return model


def get_discriminator(filters=64, kernel_initializer=kernel_init, num_downsampling=3):
    img_input = layers.Input(shape=input_img_size)
    x = layers.Conv2D(filters,(4, 4), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer)(img_input)
    x = layers.LeakyReLU(0.2)(x)

    num_filters = filters
    for num_downsample_block in range(3):
        num_filters *= 2
        if num_downsample_block < 2:
            x = downsample(x, filters=num_filters, activation=layers.LeakyReLU(0.2), kernel_size=(4, 4), strides=(2, 2))
        else:
            x = downsample(x, filters=num_filters, activation=layers.LeakyReLU(0.2), kernel_size=(4, 4), strides=(1, 1))

    x = layers.Conv2D(1, (4, 4), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer)(x)

    model = keras.models.Model(inputs=img_input, outputs=x)
    return model




#%% 

#_________________________________INITIALIZATION_______________________________

input_img_size = (256,256,1)

# Get the generators
gen_G = get_resnet_generator()
gen_F = get_resnet_generator()

# Get the discriminators
disc_X = get_discriminator()
disc_Y = get_discriminator()

# Plot G e D architectures
tf.keras.utils.plot_model(gen_G, show_shapes=True, dpi=64)
tf.keras.utils.plot_model(disc_X, show_shapes=True, dpi=64)

# Define the the optimizers for the generator and the discriminator
LR_gen = 2e-4  # Learning rate for the generator
LR_disc = 2e-4 # Learning rate for the discriminator
generator_g_optimizer = tf.keras.optimizers.Adam(learning_rate = LR_gen, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(learning_rate = LR_gen, beta_1=0.5)
discriminator_x_optimizer = tf.keras.optimizers.Adam(learning_rate = LR_disc, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(learning_rate = LR_disc, beta_1=0.5)


#%%

#___________________________CUSTOM LOSS COMPONENTS_____________________________


adv_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def gen_loss_fn(fake):
    fake_loss = adv_loss(tf.ones_like(fake), fake)
    return fake_loss

def disc_loss_fn(real, fake):
    real_loss = adv_loss(tf.ones_like(real), real)
    fake_loss = adv_loss(tf.zeros_like(fake), fake)
    return (real_loss + fake_loss) * 0.5
    
def cycle_loss(real_image, cycled_image): 
    cycle = tf.reduce_mean(tf.abs(real_image - cycled_image))
    return cycle

def id_loss(true, pred):
    idloss = tf.reduce_mean(tf.abs(pred-true))
    return idloss

# gradient loss
def gradient_loss(input_real, prediction):
    G_prediction_x, G_prediction_y = tf.image.image_gradients(prediction)
    G_target_x, G_target_y = tf.image.image_gradients(input_real)
    grad_loss = tf.math.square(tf.abs(tf.abs(G_target_x)-tf.abs(G_prediction_x))) + tf.math.square(tf.abs(tf.abs(G_target_y)-tf.abs(G_prediction_y)))
    return grad_loss

# intensity loss
def int_loss(true, pred):
    int_loss = tf.reduce_mean(tf.abs(pred-true))
    return int_loss

#%%

#______________________________BUFFER - POOL___________________________________

# update image pool for fake images to reduce model oscillation
# update discriminators using a history of generated images rather than the ones produced by the latest generators.
# Original paper recommended keeping an image buffer that stores the 50 previously created images.

def update_image_buffer_and_get_image(pool, images, buffer_capacity=50):
    selected = []
	#for image in images:
    if len(pool) < buffer_capacity:
		# stock the pool
        pool.append(images)
        selected.append(images)
    elif random.randint(0,1) < 0.5:
		# use image, but don't add it to the pool
        selected.append(images)
    else:
		# replace an existing image and use replaced image
        ix = random.randint(0, len(pool)-1)
        selected.append(pool[ix])
        pool[ix] = images
    return tf.stack(selected[0])


#%% 
# __________________________________TRAIN_STEP_________________________________

                                    
def train_step(real_x, real_y, fake_X_buffer, fake_Y_buffer, lambda_cycle=10, lambda_identity=5, lambda_adv = 1, lambda_intensity = 0.4, lambda_gdl=0.4):

        # 1. Pass real images through the generators and get the generated images
        # 2. Pass the generated images back to the generators to check if we
        #    we can predict the original image from the generated image.
        # 3. Pass the generated images in 1) to the corresponding discriminators.
        # 4. Compute losses (adversarial, cycle, intensity, gradient)
        # 5. Calculate the generators total loss (adverserial + cycle + intensity + gradient)
        # 6. Calculate the discriminators loss
        # 7. Update the weights of the generators
        # 8. Update the weights of the discriminators
        # 9. Return the losses in a dictionary
        
        with tf.GradientTape(persistent=True) as tape:
            # predict fake MRI from CoMBAT
            fake_y = gen_G(real_x, training=True)
            # predict fake CoMBAT from real_MRI
            fake_x = gen_F(real_y, training=True)

            # go back to CoMBAT using the second generator: x -> y -> x
            cycled_x = gen_F(fake_y, training=True)
            # go back to MRI using the second generator: y -> x -> y
            cycled_y = gen_G(fake_x,  training=True)
            
            # Identity
            #id_x = gen_F(real_x, training = True)
            #id_y = gen_G(real_y, training = True)

            # Discriminator output
            disc_real_x = disc_X(real_x, training=True)
            fake_x = update_image_buffer_and_get_image(fake_X_buffer, fake_x)
            disc_fake_x = disc_X(fake_x, training=True)

            disc_real_y = disc_Y(real_y, training=True)
            fake_y = update_image_buffer_and_get_image(fake_Y_buffer, fake_y)
            disc_fake_y = disc_Y(fake_y, training=True)

            # Generator adverserial loss
            gen_G_loss = gen_loss_fn(disc_fake_y)
            gen_F_loss = gen_loss_fn(disc_fake_x)

            # Generator cycle loss
            cycle_loss_G = cycle_loss(real_x, cycled_x)
            cycle_loss_F = cycle_loss(real_y, cycled_y)
            total_cycle_loss = cycle_loss_G + cycle_loss_F

            # Generator identity loss
            #id_genF = id_loss(real_x, id_x)
            #id_genG = id_loss(real_y, id_y)
            
            # Generator intensity loss
            int_genG = int_loss(real_x, fake_y)
            int_genF = int_loss(real_y, fake_x)

            # Generator gradient loss
            gdl_genG = gradient_loss(real_x, fake_y)
            gdl_genF = gradient_loss(real_y, fake_x)

            # Total generator loss
            #total_loss_G = gen_G_loss*lambda_adv + total_cycle_loss*lambda_cycle + id_genG*lambda_identity
            #total_loss_F = gen_F_loss*lambda_adv + total_cycle_loss*lambda_cycle + id_genF*lambda_identity
            total_loss_G = gen_G_loss*lambda_adv + total_cycle_loss*lambda_cycle + int_genG*lambda_intensity + gdl_genG*lambda_gdl
            total_loss_F = gen_F_loss*lambda_adv + total_cycle_loss*lambda_cycle + int_genF*lambda_intensity + gdl_genF*lambda_gdl

            #gen_loss_tot = total_loss_G + total_loss_F
            
            # Discriminator loss
            disc_X_loss = disc_loss_fn(disc_real_x, disc_fake_x)
            disc_Y_loss = disc_loss_fn(disc_real_y, disc_fake_y)
 
        # Get the gradients for the generators
        grads_G = tape.gradient(total_loss_G, gen_G.trainable_variables)
        grads_F = tape.gradient(total_loss_F, gen_F.trainable_variables)

        # Get the gradients for the discriminators
        disc_X_grads = tape.gradient(disc_X_loss, disc_X.trainable_variables)
        disc_Y_grads = tape.gradient(disc_Y_loss, disc_Y.trainable_variables)

        # Update the weights of the generators
        generator_g_optimizer.apply_gradients(zip(grads_G, gen_G.trainable_variables))
        generator_f_optimizer.apply_gradients(zip(grads_F, gen_F.trainable_variables))

        # Update the weights of the discriminators
        discriminator_x_optimizer.apply_gradients(zip(disc_X_grads, disc_X.trainable_variables))
        discriminator_y_optimizer.apply_gradients(zip(disc_Y_grads, disc_Y.trainable_variables))
        
        list_losses = [disc_X_loss.numpy(), disc_Y_loss.numpy(), gen_G_loss.numpy(), gen_F_loss.numpy(), cycle_loss_G.numpy(), cycle_loss_F.numpy(), int_genG.numpy(), int_genF.numpy(), total_loss_G.numpy(), total_loss_F.numpy()]
                    
        return list_losses 


#%%
# test step is based on the idea of tracking epoch by epoch how the metrics evolve on the test set
# to have an indication of which epoch is the best. 
# Of course, it takes time...which adds to the training time...now I replaced this part with 'Metrics.py' after training is finished.

#__________________________________TEST_STEP___________________________________

def test_step(dict_results_XtoY, dict_results_YtoX, genXtoY, genYtoX, epoch):
    
    # test data stored in hdf5
    h5f_test_COMBAT = h5py.File(folder_hdf5 + '/COMBAT_dataset_CycleGAN_test.h5', 'r') # test XCAT in hdf5 (spacingz1 and 450 slices)
    h5f_test_MRI = h5py.File(folder_hdf5 + '/MRI_dataset_CycleGAN_test.h5', 'r') # test CT in hdf5 (spacingz2 and 145 slices)
    spacing_z_COMBAT = 1
    spacing_z_MRI = 3
    
    # test data as mha
    COMBAT_test_mha = glob.glob(folder_COMBATtest + 'Cropped_P50_female_resz2.mha') # test XCAT in mha (spacing2 and 145 slices according to CT test)
    MRI_test_mha =  glob.glob(folder_MRItest + 'P13_CT0EX_1_bkw_res.mha') # test CT in mha
    
    # liver mask as mha
    COMBAT_liver_mask = glob.glob(folder_COMBATtest + 'P50_liver.mha') # from cropped phantom
    MRI_liver_mask = glob.glob(folder_MRItest + 'Test_MRI_mask.mha' )  # to be modified
    
    
    # FORWARD: prediction on XCAT (to generate fake patient) with both modalities. Values in [-1000; +1100]
    prediction_COMBAT_patches = volume_build_patches(h5f_test_COMBAT, 'hdf5', genXtoY, genYtoX, folder_results_test + 'fake_MRI/COMBAT_predicted_patches_' + str(epoch) + '.mha', "COMBAT", spacing_z_COMBAT) # imageTensorPred is [348, 348, 450]
    prediction_COMBAT_full = volume_build_full(h5f_test_COMBAT, 'hdf5', genXtoY, genYtoX, folder_results_test + 'fake_MRI/COMBAT_predicted_full_' + str(epoch) + '.mha', "COMBAT", spacing_z_COMBAT)
    print(prediction_COMBAT_patches.shape)
    print(prediction_COMBAT_full.shape)
    
    # paired
    SSIM_COMBAT_patches, SSIM_COMBAT_full = [], []
    FSIM_COMBAT_patches, FSIM_COMBAT_full = [], []
    EPR_COMBAT_patches, EPR_COMBAT_full = [], []
    EGR_COMBAT_patches, EGR_COMBAT_full = [], []
    MAE_COMBAT_patches, MAE_COMBAT_full = [], []
    
    # metrics are computed on images NON-normalized (-> MAE ok)
    for i in range(0, prediction_COMBAT_patches.shape[2]):
        image_pred_patches = prediction_COMBAT_patches[:,:, i] # [348, 348, 1]
        image_pred_patches = np.expand_dims(image_pred_patches, 2)
        print(image_pred_patches.shape)
        image_pred_full = prediction_COMBAT_full[:,:, i]
        image_pred_full = np.expand_dims(image_pred_full, 2)
        print(image_pred_full.shape)
        image_gt = h5f_test_COMBAT.get("COMBAT")[i] # (348, 348)
        image_gt = merge_HU(image_gt)
        image_gt = np.expand_dims(image_gt, 2)
        image_mask = h5f_test_COMBAT.get("maskCOMBAT")[i] # (348, 348)
        
        # metrics_patches_COMBAT 
        SSIM_COMBAT_p, FSIM_COMBAT_p, EPR_COMBAT_p, EGR_COMBAT_p, MAE_COMBAT_p = metrics_paired(image_pred_patches, image_gt, image_mask)
        SSIM_COMBAT_patches.append(SSIM_COMBAT_p)
        FSIM_COMBAT_patches.append(FSIM_COMBAT_p)
        EPR_COMBAT_patches.append(EPR_COMBAT_p)
        EGR_COMBAT_patches.append(EGR_COMBAT_p)
        MAE_COMBAT_patches.append(MAE_COMBAT_p)
        
        # metrics_full_COMBAT 
        SSIM_COMBAT_f, FSIM_COMBAT_f, EPR_COMBAT_f, EGR_COMBAT_f, MAE_COMBAT_f = metrics_paired(image_pred_full, image_gt, image_mask)
        SSIM_COMBAT_full.append(SSIM_COMBAT_f)
        FSIM_COMBAT_full.append(FSIM_COMBAT_f)
        EPR_COMBAT_full.append(EPR_COMBAT_f)
        EGR_COMBAT_full.append(EGR_COMBAT_f)
        MAE_COMBAT_full.append(MAE_COMBAT_f)
    
    # FORWARD UNPAIRED
    COMBAT_true = load(COMBAT_test_mha[0]) 
    MRI_true_in = load(MRI_test_mha[0])
    MRI_true = MRI_true_in[:, :, :]
    COMBAT_liver_mask = load(COMBAT_liver_mask[0]) 
    MRI_fake_patches = volume_build_patches(COMBAT_true, 'mha', genXtoY, genYtoX, folder_results_test + 'fake_COMBAT/MRI_predicted_full_' + str(epoch) + '.mha', "COMBAT", spacing_z_COMBAT)    
    MRI_fake_full = volume_build_full(COMBAT_true, 'mha', genXtoY, genYtoX, folder_results_test + 'fake_COMBAT/MRI_predicted_full_' + str(epoch) + '.mha', "COMBAT", spacing_z_MRI)    
    
    HistCC_COMBAT_patches, liver_mean_COMBAT_patches, liver_std_COMBAT_patches = metrics_unpaired(MRI_true, MRI_fake_patches, COMBAT_liver_mask)
    HistCC_COMBAT_full, liver_mean_COMBAT_full, liver_std_COMBAT_full = metrics_unpaired(MRI_true, MRI_fake_full, COMBAT_liver_mask)
    
    # prediction on MRI (to generate fake phantom)
    prediction_MRI_patches = volume_build_patches(h5f_test_MRI, 'hdf5', genXtoY, genYtoX, folder_results_test + 'fake_COMBAT/MRI_predicted_patches_' + str(epoch) + '.mha', "MRI", spacing_z_MRI)
    prediction_MRI_full = volume_build_full(h5f_test_MRI, 'hdf5', genXtoY, genYtoX, folder_results_test + 'fake_COMBAT/MRI_predicted_full_' + str(epoch) + '.mha', "MRI", spacing_z_MRI)    
    
    SSIM_MRI_patches, SSIM_MRI_full = [], []
    FSIM_MRI_patches, FSIM_MRI_full = [], []
    EPR_MRI_patches, EPR_MRI_full = [], []
    EGR_MRI_patches, EGR_MRI_full = [], []
    MAE_MRI_patches, MAE_MRI_full = [], []
    
    for i in range(0, prediction_MRI_patches.shape[2]):
        image_pred_patches = prediction_MRI_patches[:,:, i] # [348, 348, 1]
        image_pred_patches = np.expand_dims(image_pred_patches, 2)
        image_pred_full = prediction_MRI_full[:,:, i]
        image_pred_full = np.expand_dims(image_pred_full, 2)
        image_gt = h5f_test_MRI.get("MRI")[i] # (348, 348)
        image_gt = image_gt[:,:]
        image_gt = merge_HU(image_gt)
        image_gt = np.expand_dims(image_gt, 2)
        image_mask = h5f_test_MRI.get("maskMRI")[i] # (348, 348)
        image_mask = image_mask[:,:]
        
        # metrics_patches_MRI 
        SSIM_MRI_p, FSIM_MRI_p, EPR_MRI_p, EGR_MRI_p, MAE_MRI_p = metrics_paired(image_pred_patches, image_gt, image_mask)
        SSIM_MRI_patches.append(SSIM_MRI_p)
        FSIM_MRI_patches.append(FSIM_MRI_p)
        EPR_MRI_patches.append(EPR_MRI_p)
        EGR_MRI_patches.append(EGR_MRI_p)
        MAE_MRI_patches.append(MAE_MRI_p)
        
        # metrics_full_MRI
        SSIM_MRI_f, FSIM_MRI_f, EPR_MRI_f, EGR_MRI_f, MAE_MRI_f = metrics_paired(image_pred_full, image_gt, image_mask)
        SSIM_MRI_full.append(SSIM_MRI_f)
        FSIM_MRI_full.append(FSIM_MRI_f)
        EPR_MRI_full.append(EPR_MRI_f)
        EGR_MRI_full.append(EGR_MRI_f)
        MAE_MRI_full.append(MAE_MRI_f)
    
    # BACKWARD UNPAIRED
    COMBAT_true = load(COMBAT_test_mha[0]) 
    MRI_liver_mask = load(MRI_liver_mask[0])
    COMBAT_fake_patches = volume_build_patches(MRI_true_in, 'mha', genXtoY, genYtoX, folder_results_test + 'fake_COMBAT/MRI_predicted_full_' + str(epoch) + '.mha', "MRI", spacing_z_MRI)    
    COMBAT_fake_full = volume_build_full(MRI_true_in, 'mha', genXtoY, genYtoX, folder_results_test + 'fake_COMBAT/MRI_predicted_full_' + str(epoch) + '.mha', "MRI", spacing_z_MRI)    
    
    COMBAT_fake_patches = COMBAT_fake_patches[:, :, :]
    COMBAT_fake_full = COMBAT_fake_full[:, :, :]
    MRI_liver_mask = MRI_liver_mask[:, :, :]
    
    HistCC_MRI_patches, liver_mean_MRI_patches, liver_std_MRI_patches = metrics_unpaired(COMBAT_true, COMBAT_fake_patches, MRI_liver_mask)
    HistCC_MRI_full, liver_mean_MRI_full, liver_std_MRI_full = metrics_unpaired(COMBAT_true, COMBAT_fake_full, MRI_liver_mask)
    
    # compute mean and std of liver of XCAT true and CT true
    liver_mean_COMBAT_true = np.mean(COMBAT_true[COMBAT_liver_mask==1])
    liver_mean_MRI_true = np.mean(MRI_true_in[MRI_liver_mask==1])
    liver_std_COMBAT_true = np.std(COMBAT_true[COMBAT_liver_mask==1])
    liver_std_MRI_true = np.std(MRI_true_in[MRI_liver_mask==1])
    
    # return on the dictionaries only the mean of PSNR, SSIM and RMSE on the entire test set of slices after an epoch
    # FORWARD PAIRED
    dict_results_XtoY["Patches"]["Epoch"].append(epoch), dict_results_XtoY["Full_img"]["Epoch"].append(epoch)
    dict_results_XtoY["Patches"]["SSIM"].append(np.mean(SSIM_COMBAT_patches)), dict_results_XtoY["Full_img"]["SSIM"].append(np.mean(SSIM_COMBAT_full))
    dict_results_XtoY["Patches"]["FSIM"].append(np.mean(FSIM_COMBAT_patches)), dict_results_XtoY["Full_img"]["FSIM"].append(np.mean(FSIM_COMBAT_full))
    dict_results_XtoY["Patches"]["EPR"].append(np.mean(EPR_COMBAT_patches)), dict_results_XtoY["Full_img"]["EPR"].append(np.mean(EPR_COMBAT_full))
    dict_results_XtoY["Patches"]["EGR"].append(np.mean(EGR_COMBAT_patches)), dict_results_XtoY["Full_img"]["EGR"].append(np.mean(EGR_COMBAT_full))
    dict_results_XtoY["Patches"]["MAE"].append(np.mean(MAE_COMBAT_patches)), dict_results_XtoY["Full_img"]["MAE"].append(np.mean(MAE_COMBAT_full))
    
    # FORWARD UNPAIRED
    dict_results_XtoY["Patches"]["HistCC"].append(HistCC_COMBAT_patches), dict_results_XtoY["Full_img"]["HistCC"].append(HistCC_COMBAT_full)
    dict_results_XtoY["Patches"]["Liver_mean"].append(liver_mean_COMBAT_patches), dict_results_XtoY["Full_img"]["Liver_mean"].append(liver_mean_COMBAT_full)
    dict_results_XtoY["Patches"]["Liver_std"].append(liver_std_COMBAT_patches), dict_results_XtoY["Full_img"]["Liver_std"].append(liver_std_COMBAT_full)
    
    #BACKWARD PAIRED
    dict_results_YtoX["Patches"]["Epoch"].append(epoch), dict_results_YtoX["Full_img"]["Epoch"].append(epoch)
    dict_results_YtoX["Patches"]["SSIM"].append(np.mean(SSIM_MRI_patches)), dict_results_YtoX["Full_img"]["SSIM"].append(np.mean(SSIM_MRI_full))
    dict_results_YtoX["Patches"]["FSIM"].append(np.mean(FSIM_MRI_patches)), dict_results_YtoX["Full_img"]["FSIM"].append(np.mean(FSIM_MRI_full))
    dict_results_YtoX["Patches"]["EPR"].append(np.mean(EPR_MRI_patches)), dict_results_YtoX["Full_img"]["EPR"].append(np.mean(EPR_MRI_full))
    dict_results_YtoX["Patches"]["EGR"].append(np.mean(EGR_MRI_patches)), dict_results_YtoX["Full_img"]["EGR"].append(np.mean(EGR_MRI_full))
    dict_results_YtoX["Patches"]["MAE"].append(np.mean(MAE_MRI_patches)), dict_results_YtoX["Full_img"]["MAE"].append(np.mean(MAE_MRI_full))
    
    # BACKWARD UNPAIRED
    dict_results_YtoX["Patches"]["HistCC"].append(HistCC_MRI_patches), dict_results_YtoX["Full_img"]["HistCC"].append(HistCC_MRI_full)
    dict_results_YtoX["Patches"]["Liver_mean"].append(liver_mean_MRI_patches), dict_results_YtoX["Full_img"]["Liver_mean"].append(liver_mean_MRI_full)
    dict_results_YtoX["Patches"]["Liver_std"].append(liver_std_MRI_patches), dict_results_YtoX["Full_img"]["Liver_std"].append(liver_std_MRI_full)
    
    return None


#%% 

#____________________________________TRAIN_STEP________________________________

def train(epochs):
    
    # track training loss per epochs
    results_training_per_epochs = {}
    to_track_per_epochs =["Epoch", "D_X_loss", "D_Y_loss", "G_XtoY_gan_loss", "G_YtoX_gan_loss", "cycle_X_loss", "cycle_Y_loss", "int_x_loss", "int_y_loss", "total_loss_G", "total_loss_F"]
    for item in to_track_per_epochs:
        results_training_per_epochs[item] = []
    
    # track training loss per iterations
    results_training_per_iterations = {}
    to_track_per_iterations =["Iteration", "D_X_loss", "D_Y_loss", "G_XtoY_gan_loss", "G_YtoX_gan_loss", "cycle_X_loss", "cycle_Y_loss", "int_x_loss", "int_y_loss", "total_loss_G", "total_loss_F"]
    for item in to_track_per_iterations:
        results_training_per_iterations[item] = []
    
    # track test_metrics in case in which reconstruction is made with patches or not
    test_results_XtoY = {}
    test_results_YtoX = {}
    inference_modes = ["Patches", "Full_img"]
    for item in inference_modes:
        test_results_XtoY[item] = {}
        test_results_YtoX[item] = {}
    
    # test metrics to track (both paired and unpaired)
    metrics_to_track = ["Epoch", 'SSIM', 'FSIM', 'MAE', 'EPR', 'EGR', 'HistCC', 'Liver_mean', 'Liver_std']
    for key in test_results_XtoY: 
        dic_for = test_results_XtoY[key]
        dic_back = test_results_YtoX[key]
        for item in metrics_to_track:
            dic_for[item] = []
            dic_back[item] = []
    
    # For buffer 
    fake_X_buffer = []
    fake_Y_buffer = []
    
    # Train and test hdf5
    h5f_train_COMBAT = h5py.File(folder_hdf5 + '/COMBAT_dataset_CycleGAN_train.h5', 'r')
    h5f_train_MRI = h5py.File(folder_hdf5 + '/MRI_dataset_CycleGAN_train.h5', 'r')
    h5f_test_COMBAT = h5py.File(folder_hdf5 + '/COMBAT_dataset_CycleGAN_test.h5', 'r') # test CoMBAT in hdf5 
    h5f_test_MRI = h5py.File(folder_hdf5 + '/MRI_dataset_CycleGAN_test.h5', 'r') # test MRI in hdf5 
    
    remaining_images_combat = {}
    remaining_images_mri = {}
    
    print("Starting training...")
    for epoch in range(1, epochs):
        
        print('Epoch: ', epoch)
        
        start = time.time() # Returns the number of seconds passed since the epoch started
        
        # the dataset is composed by 33k patches. Since 1000 images are seen at each epoch, 33 epochs are needed to see the entire dataset,
        # then, start again
        if (epoch==1)|(epoch==34)|(epoch==67)|(epoch==100)|(epoch==133)|(epoch==166):
            total_images_combat = len(h5f_train_COMBAT.get("COMBAT"))
            total_images_mri = len(h5f_train_MRI.get("MRI"))
            
            # randomly select 1000 unique indices
            random_indices_combat = random.sample(range(total_images_combat), 1000)
            random_indices_mri = random.sample(range(total_images_mri), 1000)
        else:
            random_indices_combat = random.sample(list(remaining_images_combat), 1000)
            random_indices_mri = random.sample(list(remaining_images_mri), 1000)
            
        print(len(list(remaining_images_combat)))
        print(len(list(remaining_images_mri)))

        iteration = 0
        for slice_combat, slice_mri in zip(random_indices_combat, random_indices_mri):
            iteration += 1
            slice_in_combat = h5f_train_COMBAT.get("COMBAT")[slice_combat]
            slice_in_mri = h5f_train_MRI.get("MRI")[slice_mri]
                       
            slice_in_combat = np.expand_dims(slice_in_combat, axis=0)
            slice_in_combat = tf.stack([slice_in_combat], 3)
            slice_in_mri = np.expand_dims(slice_in_mri, axis=0)
            slice_in_mri = tf.stack([slice_in_mri],3)
            
            list_loss_iter = train_step(slice_in_combat, slice_in_mri, fake_X_buffer, fake_Y_buffer)
            results_training_per_iterations['Iteration'].append(iteration)
            
            if (iteration%100==0):
                print(iteration)
            
            for i in range(1, len(to_track_per_iterations)):
                item = to_track_per_iterations[i]
                results_training_per_iterations[item].append(list_loss_iter[i-1])
        
        results_training_per_epochs["Epoch"].append(epoch)
        
        for i in range(1, len(to_track_per_epochs)):
            item = to_track_per_epochs[i]
            mean = np.mean(results_training_per_iterations[item][-9:])
            results_training_per_epochs[item].append(mean)
        
        # Store weights after the training of all slices for each epoch
        gen_G.save_weights(folder_weights +'generator_g_' + str(epoch) + '.h5')
        gen_F.save_weights(folder_weights+'generator_f_' + str(epoch) + '.h5')
        disc_X.save_weights(folder_weights+'discriminator_x_' + str(epoch) + '.h5')
        disc_Y.save_weights(folder_weights+'discriminator_y_' + str(epoch) + '.h5')
        
        print("Model saved")
        print('Time taken for epoch {} is {} min\n'.format(epoch, (time.time()-start)/60))
        
        # save generated volumes (both full and patches) for visual evaluation at each epoch 
        spacing_z_fakeMRI = 2
        spacing_z_fakeCOMBAT = 3
        id_COMBAT_test = [50, 71, 86]
        id_MRI_test = [9, 13, 31]
        
        #generated_fakeMRI = []
        #generated_fakeMRI_p = []
        for idx in id_COMBAT_test:
            slices = []
            
            for slice in range(0, (len(h5f_test_COMBAT["COMBAT"]))):
                if h5f_test_COMBAT['phantom_id'][slice] == idx:
                    slices.append(slice)

            volume_build_full(h5f_test_COMBAT, slices, gen_G, gen_F, folder_results_test + 'fake_MRI/' + 'MRIfake' + '_' + str(idx) + '_' + str(epoch) + '_full.mha' ,'COMBAT', spacing_z_fakeMRI)
            volume_build_patches(h5f_test_COMBAT, slices, gen_G, gen_F, folder_results_test + 'fake_MRI/'+ 'MRIfake' + '_' + str(idx) + '_' + str(epoch) + '_patches.mha', 'COMBAT', spacing_z_fakeMRI)
            #generated_fakeMRI.append(tensorMRI)
            #generated_fakeMRI_p.append(tensorMRI_p)
        
        #generated_fakeCOMBAT = []
        #generated_fakeCOMBAT_p = []
        for idx in id_MRI_test:
            slices = []
            
            for slice in range(0, (len(h5f_test_MRI["MRI"]))):
                if h5f_test_MRI['patient_mri'][slice] == idx:
                    slices.append(slice)

            volume_build_full(h5f_test_MRI, slices, gen_G, gen_F, folder_results_test +'fake_COMBAT/' + 'COMBATfake' + '_' + str(idx) + '_' + str(epoch) + '_full.mha' ,'MRI', spacing_z_fakeCOMBAT)
            volume_build_patches(h5f_test_MRI, slices, gen_G, gen_F, folder_results_test + 'fake_COMBAT/' + 'COMBATfake' + '_' + str(idx) + '_' + str(epoch) + '_patches.mha', 'MRI', spacing_z_fakeCOMBAT)
            #generated_fakeCOMBAT.append(tensorCOMBAT)
            #generated_fakeCOMBAT_p.append(tensorCOMBAT_p)
        
        
        if (epoch==1)|(epoch==34)|(epoch==67)|(epoch==100)|(epoch==133)|(epoch==166):
            remaining_images_combat = set(range(total_images_combat)) - set(random_indices_combat)
            remaining_images_mri = set(range(total_images_mri)) - set(random_indices_mri)
        else:
            remaining_images_combat = set(remaining_images_combat) - set(random_indices_combat)
            remaining_images_mri = set(remaining_images_mri) - set(random_indices_mri)
      
        # If epoch is a multiple of 5: start test step
        #if (epoch%5==0):
        #    print('Starting validation of epoch {}'.format(epoch))
        #    test_step(test_results_XtoY, test_results_YtoX, gen_G, gen_F, epoch)
        #    plot_metrics(test_results_XtoY, test_results_YtoX, epoch)
            
        plot_losses(results_training_per_epochs, epoch)
    #return results_training_per_epochs, results_training_per_iterations, test_results_XtoY, test_results_YtoX 
    return results_training_per_epochs, results_training_per_iterations


#%%
#________________________TRAINING | TEST_______________________

EPOCHS = 150 #to be modified (150k iterations)
results_loss_train_ep, results_loss_train_iter, test_results_G, test_results_F = train(EPOCHS)

# !!! remember to save the resulting metrics in csv (not only plots!!)

#%%

# def plot_metrics(dict_for, dict_back, epoch):
    
#     cont = 0
#     for dictionary in [dict_for, dict_back]: 
#         cont +=1
#         fig, axs = plt.subplots(2, 4)
#         axs[0, 0].plot(dictionary["Patches"]["Epoch"], dictionary["Patches"]["SSIM"], color = "lightblue")
#         axs[0, 0].plot(dictionary["Full_img"]["Epoch"], dictionary["Full_img"]["SSIM"], color = "lightgreen")
#         axs[0, 0].set_title('SSIM_forward')
#         axs[0, 1].plot(dictionary["Patches"]["Epoch"], dictionary["Patches"]["FSIM"], color = "lightblue")
#         axs[0, 1].plot(dictionary["Full_img"]["Epoch"], dictionary["Full_img"]["FSIM"], color = "lightgreen")
#         axs[0, 1].set_title('FSIM_forward')
#         axs[0, 2].plot(dictionary["Patches"]["Epoch"], dictionary["Patches"]["EPR"], color = "lightblue")
#         axs[0, 2].plot(dictionary["Full_img"]["Epoch"], dictionary["Full_img"]["EPR"], color = "lightgreen")
#         axs[0, 2].set_title('EPR_forward')
#         axs[0, 3].plot(dictionary["Patches"]["Epoch"], dictionary["Patches"]["EGR"], color = "lightblue")
#         axs[0, 3].plot(dictionary["Full_img"]["Epoch"], dictionary["Full_img"]["EGR"], color = "lightgreen")
#         axs[0, 3].set_title('EGR_forward')
#         axs[1, 0].plot(dictionary["Patches"]["Epoch"], dictionary["Patches"]["MAE"], color = "lightblue")
#         axs[1, 0].plot(dictionary["Full_img"]["Epoch"], dictionary["Full_img"]["MAE"], color = "lightgreen")
#         axs[1, 0].set_title('MAE_forward')
#         axs[1, 1].plot(dictionary["Patches"]["Epoch"], dictionary["Patches"]["HistCC"], color = "lightblue")
#         axs[1, 1].plot(dictionary["Full_img"]["Epoch"], dictionary["Full_img"]["HistCC"], color = "lightgreen")
#         axs[1, 1].set_title('HistCC_forward')
#         axs[1, 2].plot(dictionary["Patches"]["Epoch"], dictionary["Patches"]["Liver_mean"], color = "lightblue")
#         axs[1, 2].plot(dictionary["Full_img"]["Epoch"], dictionary["Full_img"]["Liver_mean"], color = "lightgreen")
#         axs[1, 2].set_title('Liver_mean_forward')
#         axs[1, 3].plot(dictionary["Patches"]["Epoch"], dictionary["Patches"]["Liver_std"], color = "lightblue")
#         axs[1, 3].plot(dictionary["Full_img"]["Epoch"], dictionary["Full_img"]["Liver_std"], color = "lightgreen")
#         axs[1, 3].set_title('Liver_std_forward')
    
#         for ax in axs.flat:
#             ax.set(xlabel='Epochs', ylabel='Metric Value')
    
#         # Hide x labels and tick labels for top plots and y ticks for right plots.
#         for ax in axs.flat:
#             ax.label_outer()
        
#         if cont==1:
#             fig.savefig(folder_results_test + 'metrics_curves_forward' + '_' + str(epoch) +'.png')
#         elif cont==2:
#             fig.savefig(folder_results_test + 'metrics_curves_backward' + '_' + str(epoch) +'.png')
    
#     return None
    
    
    

    
