# -*- coding: utf-8 -*-
"""
Created on Wed May 29 10:14:51 2024

@author: fanciicam

    Patient MRI pre-processing and CoMBAT phantoms pre-processing
    Also, Patient MRI train and test hdf5 are generated

    This code does all the pre-processing steps:
       - Patient MRI need to go through bias field correction, Bilateral filtering, 
         normalization with 95th percentile, histogram clipping and histogram matching
       - CoMBAT phantoms do not need normalize1, bias field correction, bilateral filtering
"""

import SimpleITK as sitk
import os
import numpy as np
import glob
from scipy.interpolate import interp1d
import logging
import h5py
import tensorflow as tf
from random import randint
import matplotlib.pyplot as plt

desired = 'MRI' # or COMBAT
which = 'train' # or test
 
#_______________________________FOLDERS_PATIENT_MRI____________________________

if desired=='MRI':
    npatches = 8    
    dim_patch = 200
    dim_stride = 32
    if which=='train':
        folder_MRI = '//NAS-CARTCAS/camagnif/tesi/Data/mri/vibe/'
        folder_MRI_denoised = '//NAS-CARTCAS/camagnif/tesi/Data/mri/vibe_denoised/'
        folder_MRI_clean = '//NAS-CARTCAS/camagnif/tesi/Data/mri/vibe_denoised_clean/'
        folder_MRI_matched = '//NAS-CARTCAS/camagnif/tesi/Data/mri/vibe_denoised_clean_matched_train/'
    elif which=='test':    
        folder_MRI = '//NAS-CARTCAS/camagnif/tesi/Data/mri/vibe_pz_test/'
        folder_MRI_denoised = '//NAS-CARTCAS/camagnif/tesi/Data/mri/vibe_pz_denoised_test/'
        folder_MRI_clean = '//NAS-CARTCAS/camagnif/tesi/Data/mri/vibe_pz_denoised_clean_test/'
        folder_MRI_matched = '//NAS-CARTCAS/camagnif/tesi/Data/mri/vibe_pz_denoised_clean_matched_test/'
        folder_mask = '//NAS-CARTCAS/camagnif/tesi/Data/mri/vibe_mask_test/'

#______________________________FOLDERS_COMBAT_PHANTOMS_________________________
elif desired=='COMBAT':
    npatches = 9 
    dim_patch = 220
    dim_stride = 64
    if which=='train':
        folder_MRI = '//NAS-CARTCAS/camagnif/tesi/Data/mri/vibe_combat/phantoms_new/phantoms_cut/VIBE/VIBE_noise/VIBE_recon/VIBE_input/'
        folder_MRI_denoised = '//NAS-CARTCAS/camagnif/tesi/Data/mri/vibe_combat/phantoms_new/phantoms_cut/VIBE/VIBE_noise/VIBE_recon/VIBE_denoised/'
        folder_MRI_clean = '//NAS-CARTCAS/camagnif/tesi/Data/mri/vibe_combat/phantoms_new/phantoms_cut/VIBE/VIBE_noise/VIBE_recon/VIBE_denoised_clean/'
        folder_MRI_matched = '//NAS-CARTCAS/camagnif/tesi/Data/mri/vibe_combat/phantoms_new/phantoms_cut/VIBE/VIBE_noise/VIBE_recon/VIBE_denoised_clean_matched/'
    elif which=='test':
        folder_MRI = '//NAS-CARTCAS/camagnif/tesi/Data/mri/vibe_combat/test_new/test_cut/VIBE/VIBE_noise/VIBE_recon/VIBE_input/'
        folder_MRI_denoised = '//NAS-CARTCAS/camagnif/tesi/Data/mri/vibe_combat/test_new/test_cut/VIBE/VIBE_noise/VIBE_recon/VIBE_denoised/'
        folder_MRI_clean = '//NAS-CARTCAS/camagnif/tesi/Data/mri/vibe_combat/test_new/test_cut/VIBE/VIBE_noise/VIBE_recon/VIBE_denoised_clean/'
        folder_MRI_matched = '//NAS-CARTCAS/camagnif/tesi/Data/mri/vibe_combat/test_new/test_cut/VIBE/VIBE_noise/VIBE_recon/VIBE_denoised_clean_matched/'
        folder_mask = '//NAS-CARTCAS/camagnif/tesi/Data/mri/vibe_combat/test_new/test_cut/VIBE/VIBE_noise/VIBE_recon/VIBE_mask/'

fileRoot = '/'
folder_hdf5 = '//NAS-CARTCAS/camagnif/tesi/Data/mri/hdf5/' # !! TO CHANGE !!

#%%

#___________________________________FUNCTIONS__________________________________

def load_sitk(fileName):
    reader = sitk.ImageFileReader()
    reader.SetImageIO ('MetaImageIO')
    reader.SetFileName(fileName)
    imageSitk = reader.Execute()

    return imageSitk
                           
def write_sitk(filename, image):
    writer = sitk.ImageFileWriter()
    writer.SetFileName(filename)
    writer.Execute(image)

def normalize1(inputImage): # Normalization between 0 and 95% percentile of intensity values
    outArray = sitk.GetArrayFromImage( inputImage )
    outArray = np.percentile(outArray,95)*(outArray-np.min(outArray))/(np.max(outArray)-np.min(outArray))
    output = sitk.GetImageFromArray( outArray )
    output.SetSpacing(inputImage.GetSpacing())
    output.SetOrigin(inputImage.GetOrigin())
    output.SetDirection(inputImage.GetDirection())
    
    return output

def bilateral_filter(inputImage):
    bfilter = sitk.BilateralImageFilter()
    bfilter.SetRangeSigma(7)
    bfilter.SetNumberOfRangeGaussianSamples(5)
    outputImage = bfilter.Execute(inputImage)
    
    return outputImage

def create_mask(inputImageSitk, threshold=20):                           
    bnfilt = sitk.BinaryThresholdImageFilter()
    bnfilt.SetUpperThreshold(threshold)
    bnfilt.SetInsideValue(0)
    bnfilt.SetOutsideValue(1)
    maskSitk = bnfilt.Execute(inputImageSitk)
    
    padfilt = sitk.ConstantPadImageFilter()             
    UpperPad = (0,0,1)                                  
    LowerPad = (0,0,1)
    padfilt.SetPadUpperBound(UpperPad)
    padfilt.SetPadLowerBound(LowerPad)
    padfilt.SetConstant(1)
    maskSitk = padfilt.Execute(maskSitk)
    
    fhfilt = sitk.BinaryFillholeImageFilter()
    maskSitk = fhfilt.Execute(maskSitk)
    
    maskSitk = sitk.Crop(maskSitk, LowerPad, UpperPad)
    
    befilt = sitk.BinaryErodeImageFilter()
    befilt.SetForegroundValue(1)
    befilt.SetBackgroundValue(0)
    befilt.SetKernelRadius(2)
    befilt.SetKernelType(sitk.sitkBall)
    maskSitk = befilt.Execute(maskSitk)
    
    return maskSitk

def bias_correct(inputImage, maskImage):   
    corrector = sitk.N4BiasFieldCorrectionImageFilter() #https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1N4BiasFieldCorrectionImageFilter.html#details
    corrector.SetMaximumNumberOfIterations( [ 30 ] * 4  )
    corrector.SetBiasFieldFullWidthAtHalfMaximum(1)
    #corrector.SetWienerFilterNoise(0.1)
    #corrector.SetNumberOfHistogramBins(400)
    inputImage = sitk.Cast( inputImage, sitk.sitkFloat32 )
    maskImage = sitk.Cast( maskImage, sitk.sitkUInt8 )
    outSitk = corrector.Execute( inputImage, maskImage )
    
    return outSitk

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

logger = logging.getLogger(__name__)

def nyul_normalize(img_dir, mask_dir=None, output_dir=None, standard_hist=None, write_to_disk=True):
    """
    Use Nyul and Udupa method ([1,2]) to normalize the intensities of a set of MR images
    Args:
        img_dir (str): directory containing MR images
        mask_dir (str): directory containing masks for MR images
        output_dir (str): directory to save images if you do not want them saved in
            same directory as data_dir
        standard_hist (str): path to output or use standard histogram landmarks
        write_to_disk (bool): write the normalized data to disk or nah
    Returns:
        normalized (np.ndarray): last normalized image from img_dir
    References:
        [1] N. Laszlo G and J. K. Udupa, “On Standardizing the MR Image
            Intensity Scale,” Magn. Reson. Med., vol. 42, pp. 1072–1081,
            1999.
        [2] M. Shah, Y. Xiao, N. Subbanna, S. Francis, D. L. Arnold,
            D. L. Collins, and T. Arbel, “Evaluating intensity
            normalization on MRIs of human brain with multiple sclerosis,”
            Med. Image Anal., vol. 15, no. 2, pp. 267–282, 2011.
    """
    input_files = listdir_fullpath(img_dir)
    
    if output_dir is None:
        out_fns = [None] * len(input_files)
    else:
        out_fns = []
        for fn in input_files:
            base, ext = os.path.splitext(fn)
            file = os.path.basename(fn)[0:-4]
            out_fns.append(os.path.join(output_dir, file + '_hm' + ext))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            
            
    mask_files = listdir_fullpath(mask_dir)


    if standard_hist is None:
        logger.info('Learning standard scale for the set of images')
        standard_scale, percs = train(input_files, mask_files)
    elif not os.path.isfile(standard_hist):
        logger.info('Learning standard scale for the set of images')
        standard_scale, percs = train(input_files, mask_files)
        np.save(standard_hist, np.vstack((standard_scale, percs)))
    else:
        logger.info('Loading standard scale ({}) for the set of images'.format(standard_hist))
        standard_scale, percs = np.load(standard_hist)

    normalized = None
    for i, (img_fn, mask_fn, out_fn) in enumerate(zip(input_files, mask_files, out_fns)):
        base, _ = os.path.splitext(img_fn)
        logger.info('Transforming image {} to standard scale ({:d}/{:d})'.format(base, i+1, len(input_files)))
        img = load_sitk(img_fn)
        mask = load_sitk(mask_fn) if mask_fn is not None else None
        normalized = do_hist_norm(img, percs, standard_scale, mask)
        if write_to_disk:
            sitk.WriteImage(normalized, out_fn)

    return normalized, standard_scale, percs


def get_landmarks(img, percs):
    """
    get the landmarks for the Nyul and Udupa norm method for a specific image
    Args:
        img (np.ndarray): image on which to find landmarks
        percs (np.ndarray): corresponding landmark percentiles to extract
    Returns:
        landmarks (np.ndarray): intensity values corresponding to percs in img
    """
    landmarks = np.percentile(img, percs)
    return landmarks


def train(img_fns, mask_fns, i_min=1, i_max=99, i_s_min=1, i_s_max=100, l_percentile=10, u_percentile=90, step=10):
    """
    determine the standard scale for the set of images
    Args:
        img_fns (list): set of NifTI MR image paths which are to be normalized
        mask_fns (list): set of corresponding masks (if not provided, estimated)
        i_min (float): minimum percentile to consider in the images
        i_max (float): maximum percentile to consider in the images
        i_s_min (float): minimum percentile on the standard scale
        i_s_max (float): maximum percentile on the standard scale
        l_percentile (int): middle percentile lower bound (e.g., for deciles 10)
        u_percentile (int): middle percentile upper bound (e.g., for deciles 90)
        step (int): step for middle percentiles (e.g., for deciles 10)
    Returns:
        standard_scale (np.ndarray): average landmark intensity for images
        percs (np.ndarray): array of all percentiles used
    """
    percs = np.concatenate(([i_min], np.arange(l_percentile, u_percentile+1, step), [i_max]))
    standard_scale = np.zeros(len(percs))
    for i, (img_fn, mask_fn) in enumerate(zip(img_fns, mask_fns)):
        img_data = load_sitk(img_fn)
        mask = load_sitk(mask_fn)
        img_data = sitk.GetArrayFromImage(img_data)
        mask = sitk.GetArrayFromImage(mask)
        masked = img_data[mask > 0]
        landmarks = get_landmarks(masked, percs)   #calculate percentiles only where mask = 1
        min_p = np.percentile(masked, i_min)
        max_p = np.percentile(masked, i_max)
        f = interp1d([min_p, max_p], [i_s_min, i_s_max]) 
        landmarks = np.array(f(landmarks))
        standard_scale += landmarks
    standard_scale = standard_scale / len(img_fns)
    return standard_scale, percs


def do_hist_norm(img, landmark_percs, standard_scale, mask=None):
    """
    do the Nyul and Udupa histogram normalization routine with a given set of learned landmarks
    Args:
        img (nibabel.nifti1.Nifti1Image): image on which to find landmarks
        landmark_percs (np.ndarray): corresponding landmark points of standard scale
        standard_scale (np.ndarray): landmarks on the standard scale
        mask (nibabel.nifti1.Nifti1Image): foreground mask for img
    Returns:
        normalized (nibabel.nifti1.Nifti1Image): normalized image
    """
    img_data = sitk.GetArrayFromImage(img)
    mask_data = sitk.GetArrayFromImage(mask)
    masked = img_data[mask_data > 0]
    landmarks = get_landmarks(masked, landmark_percs)
    f = interp1d(landmarks, standard_scale, fill_value='extrapolate')
    normed = f(img_data)
    risultato = sitk.GetImageFromArray(normed)
    risultato.SetSpacing(img.GetSpacing())
    risultato.SetOrigin(img.GetOrigin())
    risultato.SetDirection(img.GetDirection()) 
    return risultato

#%% 

NameVIBE = glob.glob(folder_MRI + fileRoot + '*.mha')
print('Total n° of patient ' + desired + ' volumes:', np.size(NameVIBE))

#__________________________________COMPUTE_95PERCENTILE________________________

somma = 0

for i in range(0, len(NameVIBE)):
    VIBE = load_sitk(NameVIBE[i])
    outArray = sitk.GetArrayFromImage(VIBE) 
    outArray = np.percentile(outArray,95)
    somma = somma + outArray

mean = somma / len(NameVIBE)
mean = round(mean, 2)
print(mean)

#%%

#________________________________CREATE_MASKS_COMBAT___________________________

if desired == "COMBAT":
    for phantom in os.listdir(folder_MRI):
        print(phantom)
        directory_path = os.path.join(folder_MRI, phantom)
        if directory_path[-3:]=='mha': 
            print(directory_path)
            parts = phantom.split('_')
            id_ph = parts[0]
            range = "-900,980"
            output_mha_path = folder_mask + id_ph +'_mask.mha'
            cmd = f"plastimatch threshold --input {directory_path} --output {output_mha_path} --range {range}"
            os.system(cmd)
            print(('mask creation done: '+ id_ph))

#%%
#____________________________________PRE-PROCESSING____________________________

def normalize2(inputImage, mean): # Normalization between 0 and 95% percentile of intensity values. 
    outArray = sitk.GetArrayFromImage( inputImage )
    outArray = mean*(outArray-np.min(outArray))/(np.max(outArray)-np.min(outArray)) # 195.57 is the average of all 95 percentiles, for me 180.2
    output = sitk.GetImageFromArray( outArray )
    output.SetSpacing(inputImage.GetSpacing())
    output.SetOrigin(inputImage.GetOrigin())
    output.SetDirection(inputImage.GetDirection())
    
    return output

# bias field correction, normalization, bilateral filtering
# those pre-processed volumes are saved as VIBE_denoised

for i in range(0,len(NameVIBE)):
    VIBE_1 = load_sitk(NameVIBE[i])
    
    if desired=='MRI':
        VIBE = normalize1(VIBE_1)
        Maschera= create_mask(VIBE, threshold=20) 
        VIBE = bias_correct(VIBE, Maschera)
        
    VIBE = normalize2(VIBE_1, mean)                    
    
    if desired=='MRI': 
        VIBE = bilateral_filter(VIBE)

    VIBE.SetSpacing(VIBE_1.GetSpacing())
    VIBE.SetOrigin(VIBE_1.GetOrigin())
    VIBE.SetDirection(VIBE_1.GetDirection()) 

    write_sitk((folder_MRI_denoised+os.path.basename(NameVIBE[i])[0:-4]+'_denoised.mha'), VIBE)
    if desired=='mri':
        write_sitk((folder_MRI_denoised+os.path.basename(NameVIBE[i])[0:-4]+'_bias_mask.mha'), Maschera)


#%% 

#___________________HISTOGRAM CLIPPING + BACKGROUND TO 0_______________________

NameVIBE_denoised = glob.glob(folder_MRI_denoised + fileRoot + '*.mha')
Masks_VIBE = glob.glob(folder_mask + fileRoot + '*.mha')

print(len(NameVIBE_denoised))
print(len(Masks_VIBE))

for i in range (len(NameVIBE_denoised)):
    VIBE = load_sitk(NameVIBE_denoised[i])
    Seg = load_sitk(Masks_VIBE[i])

    D = VIBE.GetDepth()
    H = VIBE.GetHeight()
    W = VIBE.GetWidth()
    
    print(W,D,H)

    vibe = sitk.GetArrayFromImage(VIBE)
    seg = sitk.GetArrayFromImage(Seg)
    print(seg.shape)
    perc = np.percentile(vibe, 99)
    
    for z in range(W):
        for k in range(D):
            for j in range(H):
                 if vibe[k,j,z]>perc:
                    vibe[k,j,z] = perc


    for z in range(W):
        for k in range(D):
            for j in range(H):
                if seg[k,j,z]==0:
                    vibe[k,j,z] = 0

    Out = sitk.GetImageFromArray(vibe)
    Out.SetSpacing(VIBE.GetSpacing())
    Out.SetOrigin(VIBE.GetOrigin())
    Out.SetDirection(VIBE.GetDirection()) 
    
    write_sitk((folder_MRI_clean +os.path.basename(NameVIBE_denoised[i])[0:-4]+'_clean99.mha'), Out)

#%%

#_________________________________HISTOGRAM_MATCHING___________________________

normalized, standard_scale, percs = nyul_normalize(folder_MRI_clean , folder_mask , output_dir=folder_MRI_matched , standard_hist=None, write_to_disk=True)


#%%

#______________________________FUNCTIONS FOR HDF5 CREATION_____________________
# after having pre-processed all the volumes, create training and test hdf5

RESIZE_DIM = 256

def load(fileName):
    reader = sitk.ImageFileReader() # reads an image file and returns an SItkimage
    reader.SetImageIO("MetaImageIO")
    reader.SetFileName(fileName)
    imageSitk = reader.Execute()
    imageNp = sitk.GetArrayViewFromImage(imageSitk) #320 x 260 x [slices]
    imageNp = np.moveaxis(imageNp, 0, 2) #260 x 320 x [slices]
    tensorImg = tf.convert_to_tensor(imageNp, dtype=tf.float32)

    return tensorImg

def normalize(input_image):
    input_image = 2*((input_image-np.min(input_image)) / (np.max(input_image)-np.min(input_image))) - 1
    return input_image

# extract from volume the slice i 
def extract_slice(volume, i):
    slice_out = tf.slice(volume, [0, 0, i], [volume.shape[0], volume.shape[1], 1])
    return slice_out

def from_img_to_patch(slice_input, npatches, dim_patch, dim_stride):
    patches = tf.image.extract_patches(images=slice_input,
                               sizes=[1, dim_patch, dim_patch, 1], strides=[1, dim_stride, dim_stride, 1], # OR 220,220
                               rates=[1, 1, 1, 1], padding='VALID')
    patches_all = tf.reshape(patches, shape = (npatches, dim_patch, dim_patch, 1))

    return patches_all

def initialise_hdf5(MRIpatch_t, id_MRI_train, scan, i, desired, which, MRImask = None):
    
    # Initialise the datasets into the hdf5 file using the first slice
    h5f.create_dataset(desired, data=MRIpatch_t, compression="gzip", chunks=True, maxshape=(None, np.shape(MRIpatch_t)[1], np.shape(MRIpatch_t)[2])) #LDCT
    
    if desired == "COMBAT":
        h5f.create_dataset('phantom_id', data=[int(id_MRI_train)], compression="gzip", dtype='i', chunks=(1,), maxshape=(None,)) # patient i
    else:
        h5f.create_dataset('patient_mri', data=[int(id_MRI_train)], compression="gzip", dtype='i', chunks=(1,), maxshape=(None,)) # patient i
    
    h5f.create_dataset('scan', data=[int(scan)], compression="gzip", dtype='i', chunks=(1,), maxshape=(None,)) # patient i
    h5f.create_dataset('order', data=[i], compression="gzip", dtype='i', chunks=(1,), maxshape=(None,))  # slice number inside the single VOLUME [0-nslice_vol]
    
    if which == 'test':
        if desired == "COMBAT":
            h5f.create_dataset('COMBATmask', data=MRImask, compression="gzip", chunks=True, maxshape=(None, np.shape(MRImask)[1], np.shape(MRImask)[2])) #LDCT
        else:
            h5f.create_dataset('MRI_mask', data=MRImask, compression="gzip", chunks=True, maxshape=(None, np.shape(MRImask)[1], np.shape(MRImask)[2])) #LDCT
            
    return None

def stack_n_save(id_pz, FILE, MRI_patch, scan, i, desired, which, MRI_mask = None):
    
    MRI = tf.transpose(MRI_patch, perm=[2, 0, 1])
    
    if which == 'test':
        MRI_mask = tf.transpose(MRI_mask, perm=[2, 0, 1])

    with h5py.File(folder_hdf5 + FILE, 'a') as hf:
        hf[desired].resize((hf[desired].shape[0] + MRI.shape[0]), axis=0)
        hf[desired][-MRI.shape[0]:] = MRI

        if desired == "MRI":
            hf["patient_mri"].resize((hf["patient_mri"].shape[0] + 1), axis=0)
            hf["patient_mri"][-1:] = id_pz
        else: 
            hf["phantom_id"].resize((hf["phantom_id"].shape[0] + 1), axis=0)
            hf["phantom_id"][-1:] = id_pz
        
        if which == "test":
            if desired == "MRI":
                hf["MRI_mask"].resize((hf["MRI_mask"].shape[0] + MRI_mask.shape[0]), axis=0)
                hf["MRI_mask"][-MRI_mask.shape[0]:] = MRI_mask
            else:
                hf["COMBATmask"].resize((hf["COMBATmask"].shape[0] + MRI_mask.shape[0]), axis=0)
                hf["COMBATmask"][-MRI_mask.shape[0]:] = MRI_mask
            
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
#______________________________CREATE_TRAINING_DATASET_________________________

# extract patches and save in hdf5 file for training

if which == "train":
    
    print("CREATING THE TRAINING DATASET FOR " + desired)

    h5f = h5py.File(folder_hdf5 + desired + '_dataset_CycleGAN_train.h5', 'w')
    
    scan = os.listdir(folder_MRI_matched) 
    vol =0
    for MRI in scan:
        print(MRI)
        dir_mri = os.path.join(folder_MRI_matched, MRI)
        print(dir_mri)

        volume_pt = load(dir_mri)
        vol +=1
        
        # volume pre-processing which consist in normalize
        volume_pt = normalize(volume_pt)
        if desired == "MRI":
            id_MRI = MRI[1:3]
        else: 
            parts = MRI.split("_")
            id_MRI = parts[0]
        print(id_MRI)
    
        # iterate on the slice of each volume
        for slice in range(0, volume_pt.shape[2]):
            cont =0
            if (slice==0)&(vol==1):
                MRI_slice = extract_slice(volume_pt, slice)
                
                # extract a set of patches of dimension 200x200 from each slice
                MRI_slice = tf.expand_dims(MRI_slice, 0)
                MRI_patches = from_img_to_patch(MRI_slice, npatches, dim_patch, dim_stride) # 8 200x200 patches are extracted
                # select one random patch among the extracted ones
                vector = list(range(0, npatches)) #[0, 1, 2, 3, 4, 5, 6, 7]
                index = randint(0, len(vector)-1)
                print(index)
                pos = vector[index]
                print(pos)
                MRI_patch_t = tf.transpose(resize(MRI_patches[pos]), perm=[2, 0, 1])
                initialise_hdf5(MRI_patch_t, id_MRI, vol, slice, desired, which)
                vector.remove(pos)
                print(vector)
                for k in range(0,4):
                    index = randint(0, len(vector)-1)
                    print(index)
                    pos = vector[index]
                    print(pos)
                    stack_n_save(id_MRI, desired + '_dataset_CycleGAN_train.h5', resize(MRI_patches[pos]), vol, slice, desired, which)
                    vector.remove(pos)
                    print(vector)
                    k+=1
                    
            else:
                MRI_slice = extract_slice(volume_pt, slice)

                MRI_slice = tf.expand_dims(MRI_slice, 0)
                MRI_patches = from_img_to_patch(MRI_slice, npatches, dim_patch, dim_stride)
                
                vector = list(range(0, npatches))
                index = randint(0, len(vector)-1)
                pos = vector[index]
                #plt.imshow(MRI_patches[pos])
                #plt.show()
                stack_n_save(id_MRI, desired + '_dataset_CycleGAN_train.h5', resize(MRI_patches[pos]), vol, slice, desired, which)
                vector.remove(pos)
                for k in range(0,4):
                    index = randint(0, len(vector)-1)
                    pos = vector[index]
                    stack_n_save(id_MRI, desired + '_dataset_CycleGAN_train.h5', resize(MRI_patches[pos]), vol, slice, desired, which)
                    vector.remove(pos)
                    k +=1

#%% 

#______________________________CREATE_TESTING_DATASET__________________________
# testing images are stacked in hdf5 file with their original dimensions (no patch extraction)
# Patient test hdf5 useful only if given the patient we want to use gen_F to generate its CoMBAT version

if which == "test":
    
    print("CREATING THE TESTING DATASET FOR " + desired)

    h5f = h5py.File(folder_hdf5 + desired + '_dataset_CycleGAN_test.h5', 'w')
    
    scan = os.listdir(folder_MRI) 
    mask = os.listdir(folder_mask)
    vol =0
    for MRI in scan:
        print(MRI)
        dir_mri = os.path.join(folder_MRI, MRI)
        print(dir_mri)
        if desired == "MRI":
            id_mri = MRI[1:3]
            maskName = glob.glob(folder_mask + 'P' + id_mri + '_mask.mha')
        else: 
            parts = MRI.split("_")
            id_mri = parts[0]
            maskName = glob.glob(folder_mask + id_mri + '_mask.mha')
        
        print(maskName)

        volume_pt = load(dir_mri)
        volume_mk = load(maskName[0])
        vol +=1
        
        # volume pre-processing which consist in normalize
        volume_pt = normalize(volume_pt)
        
        # iterate on the slice of each volume
        cont=0
        for slice in range(0, volume_pt.shape[2]):
            
            if (cont==0)&(vol==1):
                MRI_slice = extract_slice(volume_pt, slice)
                MRI_mask = extract_slice(volume_mk, slice)
                    
                MRI_t = tf.transpose(MRI_slice, perm=[2, 0, 1])
                MRI_mask_t = tf.transpose(MRI_mask, perm=[2, 0, 1])
                initialise_hdf5(MRI_t, id_mri, vol, slice, desired, which, MRI_mask_t)
            else:
                cont +=1
                MRI_slice = extract_slice(volume_pt, slice)
                MRI_mask = extract_slice(volume_mk, slice)

                stack_n_save(id_mri, desired + '_dataset_CycleGAN_test.h5', MRI_slice, vol, slice, desired, which, MRI_mask)
            cont +=1



#%% Have a look to the data stacked in the hdf5 file

#______________________IMPORT MRI PATIENT DATASET AND CHECK____________________

file_to_check = desired + '_dataset_CycleGAN_' + which + '.h5' # MRI or CoMBAT/train or test

h5f = h5py.File(folder_hdf5 + file_to_check, 'r')
list(h5f.keys())
print(which +' 2D', desired, 'slices:', len(h5f.get(desired)))

# Extract data from the HDF5 file
nslices_to_viz = 4
slices_list = []

for i in range(nslices_to_viz):
    rand_slice = randint(0, len(h5f.get(desired))-1)
    new_slice = h5f.get(desired)[rand_slice]
    slices_list.append(new_slice)

plt.figure(figsize=(10, 10))
title = [desired]
for i in range(nslices_to_viz):
    plt.subplot(1, nslices_to_viz, i+1) 
    plt.title(title[0])
    plt.imshow(slices_list[i], cmap='gray')
    plt.axis('off')
plt.show()



#%%
#_________________________HOW PATCHES ARE EXTRACTED____________________________

folder_to_check = '' #!!CHANGE!! # pz MRI or CoMBAT phantom

MRI = glob.glob((folder_to_check + '.mha')) # !!SPECIFY NAME VOLUME TO CHECK!! 
MRI = load(MRI[0])
index = randint(0, MRI.shape[2])
MRI_slice = extract_slice(MRI, index)
MRI_slice = normalize(MRI_slice)
sliceMRI = tf.expand_dims(MRI_slice, 0)

size=200

plt.figure(figsize=(4, 4))
image =MRI_slice
plt.imshow(image)
plt.axis("off")

stride = 32
@tf.function
def extract_patches_inverse(shape, patches):
    _x = tf.zeros(shape)
    _y = tf.image.extract_patches(images=_x,
                                sizes=[1, 200, 200, 1],
                                strides=[1, stride, stride, 1],
                                rates=[1, 1, 1, 1],
                                padding = 'VALID')
    grad = tf.gradients(_y, _x)[0]
    # Check to avoid division by 0
    grad = tf.where(tf.abs(grad) < 1e-7, tf.ones_like(grad), grad)
    # Check to avoid NaN in output
    output = tf.gradients(_y, _x, grad_ys=patches)[0] / grad
    output = tf.where(tf.math.is_nan(output), tf.zeros_like(output), output)
    return output

def add_padding(image, pad_size):
    return tf.pad(image, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], mode='CONSTANT', constant_values=-1)

def remove_padding(image, pad_size):
    return image[:, pad_size:-pad_size, pad_size:-pad_size, :]

#pad_size = (size - stride) // 2  # padding size
#padded_sliceMRI = add_padding(sliceMRI, pad_size)

# extract patches
prova = tf.image.extract_patches(images=sliceMRI,
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

# combine patches to reconstruct image
images_reconstructed = extract_patches_inverse((1, sliceMRI.shape[1], sliceMRI.shape[2], 1), prova)
#images_n = remove_padding(images_reconstructed, pad_size)

# check that reconstruction is correct
plt.imshow(np.squeeze(images_reconstructed))
plt.axis("off")
plt.show()
