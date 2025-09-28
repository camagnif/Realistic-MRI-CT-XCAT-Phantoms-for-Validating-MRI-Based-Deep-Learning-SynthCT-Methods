# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 22:58:14 2024

@author: Bio-deib
"""

''' Pre processing CT volumes and hdf5 construction '''

#________________________LIBRARIES_AND_FOLDERS_________________________________

import SimpleITK as sitk
import numpy as np
import tensorflow as tf
import os
import glob
import cv2
import h5py 
from random import randint 
import matplotlib.pyplot as plt

folderCTin = '//NAS-CARTCAS/camagnif/tesi/Data/CT/'
folderCTout = '//NAS-CARTCAS/camagnif/tesi/Data/CT_processed/'
folderCTmask = '//NAS-CARTCAS/camagnif/tesi/Data/CT_mask/'
folderCTres = '//NAS-CARTCAS/camagnif/tesi/Data/CT_processed_res/'
folderCTmaskres = '//NAS-CARTCAS/camagnif/tesi/Data/CT_mask_res/'
folderCTtest = '//NAS-CARTCAS/camagnif/tesi/Data/CT_test/'
folderCTtestmask = '//NAS-CARTCAS/camagnif/tesi/Data/CT_test_mask/'
folder_hdf5 = '//NAS-CARTCAS/camagnif/tesi/hdf5/'
fileRoot = '/'

CREATE_TRAINING_DATASET = 1
CREATE_TESTING_DATASET = 1
RESIZE_DIM = 256

#%%

#_________________________________FUNCTIONS____________________________________

def load(fileName):
    reader = sitk.ImageFileReader() # reads an image file and returns an SItkimage
    reader.SetImageIO("MetaImageIO")
    reader.SetFileName(fileName)
    imageSitk = reader.Execute()
    imageNp = sitk.GetArrayViewFromImage(imageSitk) #320 x 260 x [slices]
    imageNp = np.moveaxis(imageNp, 0, 2) #260 x 320 x [slices]
    tensorImg = tf.convert_to_tensor(imageNp, dtype=tf.float32)

    return tensorImg

def write_sitk(filename, image):
    writer = sitk.ImageFileWriter()
    writer.SetImageIO("MetaImageIO")
    writer.SetFileName(filename)
    writer.Execute(image)
    
    return None

def air_mask(inputLDCT):
    THR_AIR = -100
    inpArray = inputLDCT.numpy()
    maskAir = inpArray < THR_AIR
    maskAir = tf.convert_to_tensor(maskAir.astype(int), dtype=tf.float32)

    return maskAir

def fill_holes(inputImage):
    im = sitk.BinaryFillholeImageFilter()
    outputImage = im.Execute(inputImage)
    
    return outputImage

def flip_axes(inputImage):
    im = sitk.FlipImageFilter()
    im.SetFlipAxes([False, True, True])
    outputImage = im.Execute(inputImage)
    return outputImage

def sobel_filter(inputImage):
    sfilter = sitk.SobelEdgeDetectionImageFilter()
    outputImage = sfilter.Execute(inputImage)
    
    return outputImage

def normalize(input_image):
    input_image = 2*((input_image-np.min(input_image)) / (np.max(input_image)-np.min(input_image))) - 1
    return input_image

# extract from volume the slice i 
def extract_slice(volume, i):
    slice_out = tf.slice(volume, [0, 0, i], [volume.shape[0], volume.shape[1], 1])
    return slice_out

# have a look to Understand_how_patches_are_extracted to find the optimal 
# patch size (200), stride (32) and n_patches (81)
def from_img_to_patch(slice_input):
    patches = tf.image.extract_patches(images=slice_input,
                               sizes=[1, 200, 200, 1], strides=[1, 32, 32, 1],
                               rates=[1, 1, 1, 1], padding='VALID')
    patches_all = tf.reshape(patches, shape = (81, 200, 200, 1))

    return patches_all

def initialise_hdf5(CTpatch_t, CTpatch_mask_t, id_CT_train, scan, i):
    
    # Initialise the datasets into the hdf5 file using the first slice
    h5f.create_dataset('CT', data=CTpatch_t, compression="gzip", chunks=True, maxshape=(None, np.shape(CTpatch_t)[1], np.shape(CTpatch_t)[2])) #LDCT
    h5f.create_dataset('maskCT', data=CTpatch_mask_t, compression="gzip", chunks=True, maxshape=(None, np.shape(CTpatch_mask_t)[1], np.shape(CTpatch_mask_t)[2])) # Mask LDCT
    h5f.create_dataset('patient_ct', data=[int(id_CT_train)], compression="gzip", dtype='i', chunks=(1,), maxshape=(None,)) # patient i
    h5f.create_dataset('scan', data=[int(scan)], compression="gzip", dtype='i', chunks=(1,), maxshape=(None,)) # patient i
    h5f.create_dataset('order', data=[i], compression="gzip", dtype='i', chunks=(1,), maxshape=(None,))  # slice number inside the single VOLUME [0-nslice_vol]

    return None

def stack_n_save(id_pz, FILE, CT_patch, CT_mask_patch, scan, i):
    
    CT = tf.transpose(CT_patch, perm=[2, 0, 1])
    CT_mask =  tf.transpose(CT_mask_patch, perm=[2, 0, 1])

    with h5py.File(folder_hdf5 + FILE, 'a') as hf:
        hf["CT"].resize((hf["CT"].shape[0] + CT.shape[0]), axis=0)
        hf["CT"][-CT.shape[0]:] = CT

        hf["maskCT"].resize((hf["maskCT"].shape[0] + CT_mask.shape[0]), axis=0)
        hf["maskCT"][-CT_mask.shape[0]:] = CT_mask

        hf["patient_ct"].resize((hf["patient_ct"].shape[0] + 1), axis=0)
        hf["patient_ct"][-1:] = id_pz
        
        hf["scan"].resize((hf["scan"].shape[0] + 1), axis=0)
        hf["scan"][-1:] = scan

        hf["order"].resize((hf["order"].shape[0] + 1), axis=0)
        hf["order"][-1:] = i

def resize(input_image): # NO PADDING AT ALL!
    input_image = tf.image.resize(input_image, [RESIZE_DIM, RESIZE_DIM], 
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, 
                                  preserve_aspect_ratio=False)
    return input_image

#%%

#____________________________PREPROCESSING_ALL_0EXCT___________________________

# FOR EACH PATIENT, FOR 0EX CT ONLY:
# 1. open CT, 
# 2. build the body mask, save and use to put bkw to -1000, 
# 3. histogram clipping in the range [-1000; +1100]
# 4. save mask and volume pre-processed

#only 0EX
session = os.listdir(folderCTin)
for element in session:
    dir_pz = os.path.join(folderCTin, element)
    print(dir_pz)
    pz = os.listdir(dir_pz)
    for patient in pz:
        directory = os.path.join(dir_pz, patient)
        print(directory)
        scan = os.listdir(directory)
        print(scan)
        for ct in scan:
            print(ct)
            x = ct.__contains__('CT0EX')
            print(x)
            if x:
                CT = glob.glob(directory + fileRoot + ct)
                print(CT[0])
                [volume_pt, refImage] = load(CT[0]) # output is a tensor
                # extract maskAir
                mask = air_mask(volume_pt)
                mask = mask.numpy()
               
                mask[0:512,0,:]=1
                mask[0:512,511,:]=1
                mask[0,0:512,:]=1
                mask[511,0:512,:]=1
                
                # go back to a sitk image to use sitk fill holes
                mask = np.moveaxis(mask, 2, 0)
                mask_sitk = sitk.GetImageFromArray(mask)
                origin = [refImage.GetOrigin()[0], refImage.GetOrigin()[1], refImage.GetOrigin()[2]]
                mask_sitk.SetOrigin(origin)
                spacing = [refImage.GetSpacing()[0],refImage.GetSpacing()[1], refImage.GetSpacing()[2]]
                mask_sitk.SetSpacing(spacing)
                mask_sitk.SetDirection(refImage.GetDirection())
                #.Show(mask_sitk)
                
                # fill holes
                mask_sitk = sitk.Cast(mask_sitk, sitk.sitkUInt8)
                mask_sitk = fill_holes(mask_sitk)
                #sitk.Show(mask_sitk)
                mask_sitk = sitk.Cast(mask_sitk, sitk.sitkFloat32)
                mask_sitk = sobel_filter(mask_sitk)
                #sitk.Show(mask_sitk)
                
                # get the largest contour for each slice of the volume
                mask_sitk = sitk.GetArrayFromImage(mask_sitk)
                mask_sitk = np.moveaxis(mask_sitk, 0, 2)
                mask_sitk = tf.convert_to_tensor(mask_sitk, dtype=tf.float32)
                print(tf.shape(mask_sitk))
                
                max_contour = np.zeros_like(mask_sitk,dtype='uint8')
                for j in range(0, mask_sitk.shape[2]):
                    mask_slice = mask_sitk[:,:,j]
                    contour = np.zeros_like(mask_slice, dtype = 'uint8')
                    contour[mask_slice>=1]=255
                    threshold = cv2.threshold(contour, 128, 1, cv2.THRESH_BINARY)[1]
                    contours = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours = contours[0] if len(contours) == 2 else contours[1]
                    result = np.zeros_like(threshold)
                    if contours != ():
                        big_contour = max(contours, key=cv2.contourArea)
                        cv2.drawContours(result, [big_contour], 0, (255,255,255), cv2.FILLED)
                    result[result==255]=1
                    max_contour[:,:,j] = result
                
                # bkw to -1000 using the body mask and clipping between -1000 and 1100
                out = volume_pt.numpy()
                out[max_contour==0]=-1000
                out[out>1100]=1100
                out[out<-1000]=-1000
                
                out = np.moveaxis(out, 2, 0)
                out = sitk.GetImageFromArray(out)
            
                cont_sitk = np.moveaxis(max_contour, 2, 0)
                cont_sitk = sitk.GetImageFromArray(cont_sitk)
                
                origin = [refImage.GetOrigin()[0], refImage.GetOrigin()[1], refImage.GetOrigin()[2]]
                cont_sitk.SetOrigin(origin)
                out.SetOrigin(origin)
                spacing = [refImage.GetSpacing()[0],refImage.GetSpacing()[1], refImage.GetSpacing()[2]]
                cont_sitk.SetSpacing(spacing)
                out.SetSpacing(spacing)
                cont_sitk.SetDirection(refImage.GetDirection())
                out.SetDirection(refImage.GetDirection())
                #sitk.Show(cont_sitk)
                write_sitk(folderCTmask + fileRoot + CT[0][-13:-4 ] + '_' + element[8] + '_mask.mha', cont_sitk)
                write_sitk(folderCTout + fileRoot + CT[0][-13:-4] + '_' + element[8] + '_bkw.mha', out)
                
#%%            

#____________________________PREPROCESSING_ALL_CT______________________________

# FOR EACH PATIENT, FOR EACH CT SCAN AVAILABLE
# 1. open CT, 
# 2. build the body mask, save and use to put bkw to -1000, 
# 3. histogram clipping in the range [-1000; +1100]
# 4. save mask and volume pre-processed    

session = os.listdir(folderCTin)
for element in session:
    dir_pz = os.path.join(folderCTin, element)
    print(dir_pz)
    pz = os.listdir(dir_pz)
    for patient in pz:
        directory = os.path.join(dir_pz, patient)
        print(directory)
        scan = os.listdir(directory)
        print(scan)
        for ct in scan:
            print(ct)

            CT = glob.glob(directory + fileRoot + ct)
            print(CT[0])
            [volume_pt, refImage] = load(CT[0]) # output is a tensor
            # extract maskAir
            mask = air_mask(volume_pt)
            mask = mask.numpy()
           
            mask[0:512,0,:]=1
            mask[0:512,511,:]=1
            mask[0,0:512,:]=1
            mask[511,0:512,:]=1
            
            # go back to a sitk image to use sitk fill holes
            mask = np.moveaxis(mask, 2, 0)
            mask_sitk = sitk.GetImageFromArray(mask)
            origin = [refImage.GetOrigin()[0], refImage.GetOrigin()[1], refImage.GetOrigin()[2]]
            mask_sitk.SetOrigin(origin)
            spacing = [refImage.GetSpacing()[0],refImage.GetSpacing()[1], refImage.GetSpacing()[2]]
            mask_sitk.SetSpacing(spacing)
            mask_sitk.SetDirection(refImage.GetDirection())
            #.Show(mask_sitk)
            
            # fill holes
            mask_sitk = sitk.Cast(mask_sitk, sitk.sitkUInt8)
            mask_sitk = fill_holes(mask_sitk)
            #sitk.Show(mask_sitk)
            mask_sitk = sitk.Cast(mask_sitk, sitk.sitkFloat32)
            mask_sitk = sobel_filter(mask_sitk)
            #sitk.Show(mask_sitk)
            
            # get the largest contour for each slice of the volume
            mask_sitk = sitk.GetArrayFromImage(mask_sitk)
            mask_sitk = np.moveaxis(mask_sitk, 0, 2)
            mask_sitk = tf.convert_to_tensor(mask_sitk, dtype=tf.float32)
            print(tf.shape(mask_sitk))
            
            max_contour = np.zeros_like(mask_sitk,dtype='uint8')
            for j in range(0, mask_sitk.shape[2]):
                mask_slice = mask_sitk[:,:,j]
                contour = np.zeros_like(mask_slice, dtype = 'uint8')
                contour[mask_slice>=1]=255
                threshold = cv2.threshold(contour, 128, 1, cv2.THRESH_BINARY)[1]
                contours = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = contours[0] if len(contours) == 2 else contours[1]
                result = np.zeros_like(threshold)
                if contours != ():
                    big_contour = max(contours, key=cv2.contourArea)
                    cv2.drawContours(result, [big_contour], 0, (255,255,255), cv2.FILLED)
                result[result==255]=1
                max_contour[:,:,j] = result
            
            # bkw to -1000 using the body mask and clipping between -1000 and 1100
            out = volume_pt.numpy()
            out[max_contour==0]=-1000
            out[out>1100]=1100
            out[out<-1000]=-1000
            
            out = np.moveaxis(out, 2, 0)
            out = sitk.GetImageFromArray(out)
        
            cont_sitk = np.moveaxis(max_contour, 2, 0)
            cont_sitk = sitk.GetImageFromArray(cont_sitk)
            
            origin = [refImage.GetOrigin()[0], refImage.GetOrigin()[1], refImage.GetOrigin()[2]]
            cont_sitk.SetOrigin(origin)
            out.SetOrigin(origin)
            spacing = [refImage.GetSpacing()[0],refImage.GetSpacing()[1], refImage.GetSpacing()[2]]
            cont_sitk.SetSpacing(spacing)
            out.SetSpacing(spacing)
            cont_sitk.SetDirection(refImage.GetDirection())
            out.SetDirection(refImage.GetDirection())
            #sitk.Show(cont_sitk)
            write_sitk(folderCTmask + ct[:-4] + '_' + element[8] + '_mask.mha', cont_sitk)
            write_sitk(folderCTout + ct[:-4] + '_' + element[8] + '_bkw.mha', out)


#%%

#__________________________________RESAMPLE____________________________________

import subprocess

def call_plastimatch_resample(input_path, output_path, spacing_x, spacing_y, spacing_z):
    spacing_option = f"{spacing_x} {spacing_y} {spacing_z}"
    command = [
        'plastimatch', 'resample',
        '--input', input_path,
        '--output', output_path,
        '--spacing', spacing_option
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

# resample CT volumes
for pz in os.listdir(folderCTout):
    print(pz)
    directory_path = os.path.join(folderCTout, pz)
    print(directory_path)
    spacing = [1.0625, 1.0625, 2.0]
    spacing_x = spacing[0]
    spacing_y = spacing[1]
    spacing_z = spacing[2]
    output_mha_path = folderCTres + pz[:-4] +'_res.mha'
    print(output_mha_path)
    call_plastimatch_resample(directory_path, output_mha_path, spacing_x, spacing_y, spacing_z)
    print('resampling done')

# resample CT mask
for pz in os.listdir(folderCTmask):
    print(pz)
    directory_path = os.path.join(folderCTmask, pz)
    print(directory_path)
    spacing = [1.0625, 1.0625, 2.0]
    spacing_x = spacing[0]
    spacing_y = spacing[1]
    spacing_z = spacing[2]
    output_mha_path = folderCTmaskres + pz[:-4] +'_res.mha'
    print(output_mha_path)
    call_plastimatch_resample(directory_path, output_mha_path, spacing_x, spacing_y, spacing_z)
    print('resampling done')

#%%

#____________________________CREATE_TRAINING_DATASET___________________________

# extract patches and save in hdf5 file

lista_id = [13, 17, 19, 20, 23, 24, 25, 26, 27, 28, 29, 30, 31] # they need to be rotated
lista_id_second = ['01', '02']

#_____________________________CREATE TRAINING DATASET__________________________

if CREATE_TRAINING_DATASET:
    
    print("CREATING THE TRAINING DATASET")

    h5f = h5py.File(folder_hdf5 + 'CT_dataset_CycleGAN_train_again3.h5', 'w')
    
    scan = os.listdir(folderCTres) 
    mask = os.listdir(folderCTmaskres)
    
    vol =0
    for CT in scan:
        print(CT)
        dir_ct = os.path.join(folderCTres, CT)
        print(dir_ct)
        maskName = glob.glob(folderCTmaskres + str(CT[:-11]) + 'mask_res.mha')
        print(maskName)

        volume_pt = load(dir_ct) # output is a tensor 
        volume_mk = load(maskName[0])
        vol +=1
        
        # volume pre-processing which consist in normalize
        volume_pt = normalize(volume_pt)
        if vol<33:
            id_CT = CT[2:4]
        else: 
            id_CT = CT[1:3]
        print(id_CT)
        
        # iterate on the slice of each volume
        for slice in range(0, volume_pt.shape[2]):
            cont =0
            if (slice==0)&(vol==1):
                CT_slice = extract_slice(volume_pt, slice)
                CT_mask = extract_slice(volume_mk, slice)
                
                if int(id_CT) in lista_id:
                    CT_slice = tf.image.rot90(CT_slice,k=2,name=None)
                    CT_mask = tf.image.rot90(CT_mask,k=2,name=None)
                
                if (id_CT in lista_id_second)&(vol<33): 
                    CT_slice = tf.image.rot90(CT_slice,k=2,name=None)
                    CT_mask = tf.image.rot90(CT_mask,k=2,name=None)
                
                # extract a set of patches of dimension 200x200 from each slice
                CT_slice = tf.expand_dims(CT_slice, 0)
                CT_mask = tf.expand_dims(CT_mask, 0) 
                CT_patches = from_img_to_patch(CT_slice) # 8 200x200 patches are extracted
                CT_mask_patches = from_img_to_patch(CT_mask)
                    
                # select one random patch among the extracted ones
                #npatches = 8
                vector = list(range(47, 53)) #[45, 46, 47, 48, 49, 50, 51, 52, 53]
                index = randint(0, len(vector)-1)
                print(index)
                pos = vector[index]
                print(pos)
                CT_patch_t = tf.transpose(resize(CT_patches[pos]), perm=[2, 0, 1])
                CT_mask_patch_t = tf.transpose(resize(CT_mask_patches[pos]), perm=[2, 0, 1])
                initialise_hdf5(CT_patch_t, CT_mask_patch_t, id_CT, vol, slice)
                vector.remove(pos)
                print(vector)
                for k in range(0,1):
                    index = randint(0, len(vector)-1)
                    print(index)
                    pos = vector[index]
                    print(pos)
                    stack_n_save(id_CT, 'CT_dataset_CycleGAN_train_again3.h5', resize(CT_patches[pos]), resize(CT_mask_patches[pos]), vol, slice)
                    vector.remove(pos)
                    print(vector)
                    k+=1
            else:
                #print(slice)
                CT_slice = extract_slice(volume_pt, slice)
                CT_mask = extract_slice(volume_mk, slice)
                
                if int(id_CT) in lista_id:
                    CT_slice = tf.image.rot90(CT_slice,k=2,name=None)
                    CT_mask = tf.image.rot90(CT_mask,k=2,name=None)
                
                if (id_CT in lista_id_second)&(vol<33): 
                    CT_slice = tf.image.rot90(CT_slice,k=2,name=None)
                    CT_mask = tf.image.rot90(CT_mask,k=2,name=None)
                
                CT_slice = tf.expand_dims(CT_slice, 0)
                CT_mask = tf.expand_dims(CT_mask, 0) 
                CT_patches = from_img_to_patch(CT_slice)
                CT_mask_patches = from_img_to_patch(CT_mask)
                
                #npatches = 81 
                vector = list(range(47, 53))
                index = randint(0, len(vector)-1)
                pos = vector[index]
                if (slice%100==0):
                    plt.imshow(CT_patches[pos])
                    plt.show()
                stack_n_save(id_CT, 'CT_dataset_CycleGAN_train_again3.h5', resize(CT_patches[pos]), resize(CT_mask_patches[pos]), vol, slice)
                vector.remove(pos)
                for k in range(0,1):
                    index = randint(0, len(vector)-1)
                    pos = vector[index]
                    stack_n_save(id_CT, 'CT_dataset_CycleGAN_train_again3.h5', resize(CT_patches[pos]), resize(CT_mask_patches[pos]), vol, slice)
                    vector.remove(pos)
                    k +=1
            cont +=1

# Train 2D CT slices: 50268

#%% 

#_____________________________CREATE TESTING DATASET__________________________

# no patches are extracted here (prediction will be done both dividing in patches and on full img)

if CREATE_TESTING_DATASET:
    
    print("CREATING THE TESTING DATASET")

    h5f = h5py.File(folder_hdf5 + 'CT_dataset_CycleGAN_test_again2.h5', 'w')
    
    scan = os.listdir(folderCTtest) 
    mask = os.listdir(folderCTtestmask)
    vol =0
    for CT in scan:
        print(CT)
        dir_ct = os.path.join(folderCTtest, CT)
        print(dir_ct)
        maskName = folderCTtestmask + str(CT[:-23]) + '_bkw_res_mask.mha'
        print(maskName)

        volume_pt = load(dir_ct) # output is a tensor 
        volume_mk = load(maskName)
        vol +=1
        
        # volume pre-processing which consist in normalize
        volume_pt = normalize(volume_pt)
        id_CT = CT[1:3]
        print(id_CT)
        
        # iterate on the slice of each volume
        for slice in range(0, volume_pt.shape[2]):
            cont =0
            if (slice==0)&(vol==1):
                CT_slice = extract_slice(volume_pt, slice)
                CT_mask = extract_slice(volume_mk, slice)
                
                if int(id_CT) in lista_id:
                    CT_slice = tf.image.rot90(CT_slice,k=2,name=None)
                    CT_mask = tf.image.rot90(CT_mask,k=2,name=None)
                
                if (id_CT in lista_id_second)&(vol<33): 
                    CT_slice = tf.image.rot90(CT_slice,k=2,name=None)
                    CT_mask = tf.image.rot90(CT_mask,k=2,name=None)
                
                CT_slice = tf.transpose(CT_slice, perm=[2, 0, 1])
                CT_mask = tf.transpose(CT_mask, perm=[2, 0, 1])
                initialise_hdf5(CT_slice, CT_mask, id_CT, vol, slice)
            else:
                #print(slice)
                CT_slice = extract_slice(volume_pt, slice)
                CT_mask = extract_slice(volume_mk, slice)
                
                if int(id_CT) in lista_id:
                    CT_slice = tf.image.rot90(CT_slice,k=2,name=None)
                    CT_mask = tf.image.rot90(CT_mask,k=2,name=None)
                
                if (id_CT in lista_id_second)&(vol<33): 
                    CT_slice = tf.image.rot90(CT_slice,k=2,name=None)
                    CT_mask = tf.image.rot90(CT_mask,k=2,name=None)
                
                plt.imshow(CT_slice)
                plt.show()
                stack_n_save(id_CT, 'CT_dataset_CycleGAN_test_again2.h5', CT_slice, CT_mask, vol, slice)

# 412 slices for test

#%% Have a look to the training set stacked in the hdf5 file

#____________________________IMPORT_DATASET_AND_CHECK__________________________

desired = 'XCAT' # or XCAT
file_to_check = desired + '_dataset_CycleGAN_test.h5' # train or test

h5f = h5py.File(folder_hdf5 + file_to_check, 'r')
list(h5f.keys())
print('Train 2D', desired, 'slices:', len(h5f.get(desired)))

# Extract data from the HDF5 file
nslices_to_viz = 4
slices_list = []
mask_slices_list = []

for i in range(nslices_to_viz):
    rand_slice = randint(0, len(h5f.get(desired))-1)
    new_slice = h5f.get(desired)[rand_slice]
    slices_list.append(new_slice)
    new_mask_slice = h5f.get("mask"+desired)[rand_slice]
    mask_slices_list.append(new_mask_slice)


plt.figure(figsize=(10, 10))
title = [desired]
for i in range(nslices_to_viz):
    plt.subplot(1, nslices_to_viz, i+1) 
    plt.title(title[0])
    plt.imshow(slices_list[i], cmap='gray')
    plt.axis('off')
plt.show()


plt.figure(figsize=(10, 10))
title = ['mask'+desired]
for i in range(nslices_to_viz):
    plt.subplot(1, nslices_to_viz, i+1) 
    plt.title(title[0])
    plt.imshow(mask_slices_list[i], cmap='gray')
    plt.axis('off')
plt.show()


#%%
#_________________________HOW_PATCHES_ARE_EXTRACTED____________________________

#CT = glob.glob((folderCTres + 'P08_CT0EX_1_bkw_res.mha'))
CT = glob.glob(folderCTtest + 'P13_CT0EX_1_bkw_res_bkw_masked.mha')
CT = load(CT[0]) # output is a tensor 
index = randint(0, CT.shape[2])
CT_slice = extract_slice(CT, index)
CT_slice = normalize(CT_slice)
#CT_slice = CT_slice[46:466, 0:420]
#CT_slice = CT_slice[50:420,50:420]
#CT_slice = CT_slice[70:442,80:432]
sliceCT = tf.expand_dims(CT_slice, 0)

size=220

plt.figure(figsize=(4, 4))
image = CT_slice
plt.imshow(image)
plt.axis("off")

stride = 32
@tf.function
def extract_patches_inverse(shape, patches):
    _x = tf.zeros(shape)
    _y = tf.image.extract_patches(images=_x,
                                sizes=[1, size, size, 1],
                                strides=[1, stride, stride, 1],
                                rates=[1, 1, 1, 1],
                                padding = 'VALID')
    grad = tf.gradients(_y, _x)[0]
    # Controllo per evitare divisione per zero
    grad = tf.where(tf.abs(grad) < 1e-7, tf.ones_like(grad), grad)
    # Controllo per evitare NaN nell'output
    output = tf.gradients(_y, _x, grad_ys=patches)[0] / grad
    output = tf.where(tf.math.is_nan(output), tf.zeros_like(output), output)
    return output

# padding is needed for patch extraction
def add_padding(image, pad_size):
    return tf.pad(image, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], mode='CONSTANT', constant_values=-1)

def remove_padding(image, pad_size):
    return image[:, pad_size:-pad_size, pad_size:-pad_size, :]

#pad_size = (size - stride) // 2  # padding size
#padded_sliceCT = add_padding(sliceCT, pad_size)
padded_sliceCT = sliceCT

# extract patches
prova = tf.image.extract_patches(images=padded_sliceCT,
                                 sizes=[1, size, size, 1],
                                 strides=[1, stride, stride, 1],
                                 rates=[1, 1, 1, 1],
                                 padding = 'VALID')

# have a look to img division in patches
plt.figure(figsize=(20, 20))
for imgs in prova:
    count = 0
    for r in range(prova.shape[1]):
        for c in range(prova.shape[2]):
            ax = plt.subplot(prova.shape[1], prova.shape[2], count+1)
            plt.imshow(tf.reshape(imgs[r,c],shape=(size,size,1)).numpy(), vmin = -1, vmax = 1)
            ax.set_adjustable('box')
            count += 1
            #plt.imshow(tf.reshape(imgs[r,c],shape=(size,size,1)).numpy(), vmin = -1, vmax = 1)
            #plt.show()
            
#prova_n = tf.reshape(prova, shape = (81, 200, 200, 1))
prova_n = tf.reshape(prova, shape = (8, 220, 220, 1))
# combine patches to reconstruct image
images_reconstructed = extract_patches_inverse((1, padded_sliceCT.shape[1], padded_sliceCT.shape[2], 1), prova)
#images_n = remove_padding(images_reconstructed, pad_size)

#%%
# check that reconstruction is correct
plt.imshow(np.squeeze(images_reconstructed))
plt.axis("off")
plt.show()


