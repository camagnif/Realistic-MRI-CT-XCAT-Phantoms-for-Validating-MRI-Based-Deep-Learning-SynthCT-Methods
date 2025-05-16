# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 19:49:06 2024

@author: fanciicam

PREDICT REALISTIC MRI GIVEN COMBAT PHANTOM IN HDF5

Since the net has been trained on patches, prediction should be carried out on patches, therefore:
   1. the test image (320x260 or 348x348) is stored in the hdf5 test file
   2. it is divided in patches: patches of 220x220 with stride 32 were found to be good for both dimensions
   3. resize each patch to 256x256 for model prediction
   4. combine predicted-patches to reconstruct the full-img (patches must be resized back to 220x220 before reconstruction)
"""

#__________________________________LIBRARIES___________________________________

import tensorflow as tf
import numpy as np
import SimpleITK as sitk
from tensorflow.keras import layers
import tensorflow_addons as tfa
from tensorflow import keras
import h5py
import glob

#___________________________________FOLDERS____________________________________

desired = 'train'

if desired=='train':
    id_ph = [50, 76, 77, 80, 89, 92, 96, 98, 106, 108, 117, 118, 128, 139, 140, 141, 142, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 157, 159, 162, 163, 164, 166, 167, 168, 169, 170, 171, 173, 175, 176, 178, 180, 182, 184, 196, 200, 201]
    folder_COMBAT = '//NAS-CARTCAS/camagnif/tesi/Data/mri/vibe_combat/phantoms_new/phantoms_cut/VIBE/VIBE_noise/VIBE_recon/VIBE_denoised_clean_matched/'
    folder_mask_COMBAT = '//NAS-CARTCAS/camagnif/tesi/FINAL_RESULTS/MRI/train/train_348x348_mask/'
    folder_save = '//NAS-CARTCAS/camagnif/tesi/FINAL_RESULTS/MRI/train/train_348x348_50/'
elif desired=='test':
    id_ph = [50, 71, 86, 99, 143]
    folder_COMBAT = '//NAS-CARTCAS/camagnif/tesi/FINAL_RESULTS/MRI/test/test_originalph_348x348_5/'
    folder_mask_COMBAT = '//NAS-CARTCAS/camagnif/tesi/FINAL_RESULTS/MRI/test/test_348x348_mask/'
    folder_save = '//NAS-CARTCAS/camagnif/tesi/FINAL_RESULTS/MRI/test/test_348x348_5/'

folder_weights = '//NAS-CARTCAS/camagnif/tesi/Data/mri/weights_TRAINING_MRI_DEF/'
folder_hdf5 = '//NAS-CARTCAS/camagnif/tesi/FINAL_RESULTS/MRI/' + desired + '/hdf5_forprediction/'

#%%

#__________________________________FUNCTIONS___________________________________

RESIZE_DIM = 256

def load(fileName):
    reader = sitk.ImageFileReader() # reads an image file and returns an SItkimage
    reader.SetImageIO("MetaImageIO")
    reader.SetFileName(fileName)
    imageSitk = reader.Execute()
    imageNp = sitk.GetArrayViewFromImage(imageSitk) 
    imageNp = np.moveaxis(imageNp, 0, 2) 
    tensorImg = tf.convert_to_tensor(imageNp, dtype=tf.float32)

    return tensorImg

def normalize(input_image):
    input_image = 2*((input_image-np.min(input_image)) / (np.max(input_image)-np.min(input_image))) - 1
    return input_image

# extract from volume the slice i 
def extract_slice(volume, i):
    slice_out = tf.slice(volume, [0, 0, i], [volume.shape[0], volume.shape[1], 1])
    return slice_out

def expand_dim(input_img, axis):
    output = tf.expand_dims(input_img, axis) 
    return output

def resize(input_image):                        #NOT CHANGE: 256!
    input_image = tf.image.resize(input_image, [RESIZE_DIM, RESIZE_DIM], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, preserve_aspect_ratio=False)
    return input_image

def resize_new(input_image):                    #CHANGE
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

# function that combines the patches to reconstruct the entire img
@tf.function
def extract_patches_inverse(shape, patches, stride, size):
    _x = tf.zeros(shape)
    _y = tf.image.extract_patches(images=_x,
                                sizes=[1, size, size, 1],
                                strides=[1, stride, stride, 1],
                                rates=[1, 1, 1, 1],
                                padding = 'VALID')
    grad = tf.gradients(_y, _x)[0]
    # Check to avoid division by 0
    grad = tf.where(tf.abs(grad) < 1e-7, tf.ones_like(grad), grad)
    # Check to avoid Nan in output
    output = tf.gradients(_y, _x, grad_ys=patches)[0] / grad
    output = tf.where(tf.math.is_nan(output), tf.zeros_like(output), output)
    return output

def reconstruct_img_patches(slice_in, model):
    # slice_in arrives from hdf5 as a tensor ([348, 348] for MRI)
    size = 220 #CHANGE
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
    patch = tf.reshape(patches, shape=(n_patches, size, size, 1))
    i = 0
    patch_temp = expand_dim(patch[i], 0) #patch[i]: [220, 220, 1] patch_temp: [1, 220, 220, 1]
    tensor = expand_dim(tf.squeeze(resize_new(model(resize(patch_temp)))),0)
    for i in range(1,n_patches): # 9
        patch_temp = expand_dim(patch[i], 0) #patch[i]: [256, 256, 1]
        tensor = tf.concat([tensor, expand_dim(tf.squeeze(resize_new(model(resize(patch_temp)))),0)],0) 
        
        # since patch_temp is 220x220, we need to resize it to 256x256 for model prediction
        # model(resize(patch)) è [1, 256, 256, 1], resize new is [220x220], squeeze will be [220, 220], expand è [1, 220, 220] -> tensor will be [num_patches, 220, 220, 1]
    
    patch_new = tf.reshape(tensor, shape = (1, patches.shape[1], patches.shape[2], size*size))  
    # reconstruction function receives in input the patches organized as (1,3,3,48400). 48.400 is the num of pixels of each patch (220x220)
    images_reconstructed = extract_patches_inverse((1, slice_in.shape[1], slice_in.shape[2], 1), patch_new, stride, size)
    #images_reconstructed = remove_padding(images_reconstructed, pad_size)

    return tf.squeeze(images_reconstructed), tensor # [348, 348] for MRI

# from [-1;+1] back to original values range: [0-100] for MRI VIBE
def merge_HU(predTensor):
    ct = predTensor[:, :]
    ctnew = 0.5*(100-(0))*(ct+1) + (0)  
    return ctnew

# construction of the tensorflow tensor of the predicted volume + write of the mha volume 
def volume_build_patches(file, slices, modelXtoY, modelYtoX, synthFileName, mod, spacing_z, mask_volume, write=True):
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
            
            imageTensorPred = imageTensorPred.numpy()
            mask_volume = mask_volume.numpy()
            imageTensorPred[mask_volume==0]=0
            imageTensorPred = tf.convert_to_tensor(imageTensorPred, dtype=tf.float32) # [348, 348]
            writeSynthetic(imageTensorPred, synthFileName,  spacing_z, write=write)           
            
            count = -1  # new volume
        count = count + 1
        
    return imageTensorPred

def initialise_hdf5(COMBATpatch_t, COMBATmask_patch_t, id_COMBAT, scan, i):
    
    # Initialise the datasets into the hdf5 file using the first slice
    h5f.create_dataset('COMBAT', data=COMBATpatch_t, compression="gzip", chunks=True, maxshape=(None, np.shape(COMBATpatch_t)[1], np.shape(COMBATpatch_t)[2])) 
    h5f.create_dataset('COMBATmask', data=COMBATmask_patch_t, compression="gzip", chunks=True, maxshape=(None, np.shape(COMBATmask_patch_t)[1], np.shape(COMBATmask_patch_t)[2])) 
    h5f.create_dataset('phantom_id', data=[int(id_COMBAT)], compression="gzip", dtype='i', chunks=(1,), maxshape=(None,)) 
    h5f.create_dataset('scan', data=[int(scan)], compression="gzip", dtype='i', chunks=(1,), maxshape=(None,)) 
    h5f.create_dataset('order', data=[i], compression="gzip", dtype='i', chunks=(1,), maxshape=(None,))

    return None

def stack_n_save(id_phantom, FILE, COMBAT_patch, COMBAT_mask_patch, scan, i):
    
    COMBAT = tf.transpose(COMBAT_patch, perm=[2, 0, 1])
    COMBAT_mask =  tf.transpose(COMBAT_mask_patch, perm=[2, 0, 1])

    with h5py.File(folder_hdf5 + FILE, 'a') as hf:
        hf["COMBAT"].resize((hf["COMBAT"].shape[0] + COMBAT.shape[0]), axis=0)
        hf["COMBAT"][-COMBAT.shape[0]:] = COMBAT
        
        hf["COMBATmask"].resize((hf["COMBATmask"].shape[0] + COMBAT_mask.shape[0]), axis=0)
        hf["COMBATmask"][-COMBAT_mask.shape[0]:] = COMBAT_mask

        hf["phantom_id"].resize((hf["phantom_id"].shape[0] + 1), axis=0)
        hf["phantom_id"][-1:] = id_phantom
        
        hf["scan"].resize((hf["scan"].shape[0] + 1), axis=0)
        hf["scan"][-1:] = scan

        hf["order"].resize((hf["order"].shape[0] + 1), axis=0)
        hf["order"][-1:] = i
        
#%%

#__________________________________CYCLEGAN____________________________________

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

#%%

#_________________________________INITIALIZATION_______________________________

input_img_size = (256,256,1)

# Get the generators
gen_G = get_resnet_generator()
gen_F = get_resnet_generator()

# Plot G e D architectures
tf.keras.utils.plot_model(gen_G, show_shapes=True, dpi=64)

# Define the the optimizers for the generator and the discriminator
LR_gen = 2e-4  # Learning rate for the generator
LR_disc = 2e-4 # Learning rate for the discriminator
generator_g_optimizer = tf.keras.optimizers.Adam(learning_rate = LR_gen, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(learning_rate = LR_gen, beta_1=0.5)

# Load the best generator
best_epoch = 135 # best epoch
gen_G.load_weights(folder_weights +'/generator_g_' + str(int(best_epoch)) + '.h5')  
gen_F.load_weights(folder_weights +'/generator_f_' + str(int(best_epoch)) + '.h5')  

#%%

#___________________________HDF5_for_PREDICTION________________________________

CREATE_DATASET = 1

if CREATE_DATASET:
    
    print("CREATING THE DATASET")

    h5f = h5py.File(folder_hdf5 + 'COMBAT_dataset_CycleGAN_' + desired +'_set.h5', 'w')
    
    vol = 0
    for id in id_ph:
        vol +=1
        
        combat_name = glob.glob(folder_COMBAT + str(id) + '*.mha')
        volume_pt = load(combat_name[0]) 
        volume_pt = normalize(volume_pt)
        id_combat = id
        print(id_combat)
        
        maskName = glob.glob(folder_mask_COMBAT + str(id) + '*.mha')
        volume_mk = load(maskName[0])
        
        cont=0
        for slice in range(0, volume_pt.shape[2]):
            
            if (cont==0)&(vol==1):
                COMBAT_slice = extract_slice(volume_pt, slice)
                COMBAT_mask = extract_slice(volume_mk, slice)

                COMBAT_t = tf.transpose(COMBAT_slice, perm=[2, 0, 1])
                COMBAT_mask_t = tf.transpose(COMBAT_mask, perm=[2, 0, 1])
                initialise_hdf5(COMBAT_t, COMBAT_mask_t, id_combat, vol, slice)
            else:
                cont +=1
                COMBAT_slice = extract_slice(volume_pt, slice)
                COMBAT_mask = extract_slice(volume_mk, slice)
                stack_n_save(id_combat, 'COMBAT_dataset_CycleGAN_' + desired + '_set.h5', COMBAT_slice, COMBAT_mask, vol, slice)
            cont +=1

#%%

#_____________________________PREDICTION_______________________________________

h5f_COMBAT = h5py.File(folder_hdf5 + '/COMBAT_dataset_CycleGAN_' + desired + '_set.h5', 'r') # test COMBAT in hdf5

spacing_z_fakeMRI = 1 # 1 for train and 2 for test

for id in id_ph:
    print(id)
    slices = []
    
    for slice in range(0, (len(h5f_COMBAT["COMBAT"]))):
        if h5f_COMBAT['phantom_id'][slice] == id:
            slices.append(slice)
        
        maskName = glob.glob(folder_mask_COMBAT + str(id) + '*.mha')
        volume_mk = load(maskName[0])
        
    volume_build_patches(h5f_COMBAT, slices, gen_G, gen_F, folder_save + str(id) + '_' + 'fakeMRI' + '_' + desired + '_corr_prova.mha', 'COMBAT', spacing_z_fakeMRI, volume_mk)
    
    