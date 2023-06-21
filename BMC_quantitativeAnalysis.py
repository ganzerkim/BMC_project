# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 10:49:11 2023

@author: User
"""
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from os import listdir
from os.path import isfile, join
import os
import cv2 as cv
#import SimpleITK as sitk
import pydicom._storage_sopclass_uids

# metadata
fileMeta = pydicom.Dataset()
fileMeta.MediaStorageSOPClassUID = pydicom._storage_sopclass_uids.CTImageStorage
fileMeta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
fileMeta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

def selection_sort(dicom):
    
    for i in range(len(dicom)):
                
        for j in range(i + 1, len(dicom)):
            value = dicom[j].SliceLocation
            value_d = dicom[j]
            if value < dicom[i].SliceLocation:
                                
                temp = dicom[i]
                dicom[i] = value_d
                dicom[j] = temp
                
        print(i)
    return dicom

def img_loader(images_path):
    path_tmp = []
    name_tmp = []
    img_tmp = []

    for (path, dir, files) in os.walk(images_path):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            
            if ext == '.dcm' or '.IMA':
                print("%s/%s" % (path, filename))
                path_tmp.append(path)
                name_tmp.append(filename)

    dcm_tmp = []

    for i in range(len(path_tmp)):
        dcm_p = pydicom.dcmread(path_tmp[i] + '/' + name_tmp[i], force = True)
        dcm_tmp.append(dcm_p)
    
    r_dcm_tmp = selection_sort(dcm_tmp)
    
    for ii in range(len(r_dcm_tmp)):
        img_tmp.append(r_dcm_tmp[ii].pixel_array)
        
        
    return dcm_tmp, img_tmp


def fourier(img):
    #f = np.fft.fft2(img)
    #fshift = np.fft.fftshift(f)
    #m_spectrum = 20*np.log(np.abs(fshift))
    
    norm_image = cv.normalize(img, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    amp = (np.fft.fftshift(np.fft.fft2(norm_image)))
    amp_log = np.log(np.abs(amp))
    norm_amp = cv.normalize(amp_log, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    
    
    plt.figure()
    plt.subplot(121), plt.imshow(norm_image, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(norm_amp, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    
    #plt.show()
    
    return amp, amp_log, norm_amp

def ifourier(amp):
    restored = np.abs(np.fft.ifft2(np.fft.ifftshift(amp)))
    
    return restored

def histdraw(img, cnt, rng):
    plt.figure()
    hist,bins = np.histogram(img.ravel(), cnt, rng)
    plt.hist(img.ravel(), cnt, rng); 
    plt.show()
    
    return hist, bins
   

###################################
nomoco_img_path = 'C:/Users/User/Desktop/Project Support/Motion correction/best/BMC_BestCase/095-LM-00-PSFTOF_000_000.v 8i 5s'
moco_img_path = 'C:/Users/User/Desktop/Project Support/Motion correction/best/BMC_BestCase/095-BMC-LM-00-ac_mc_000_000.v 8i 5s'

nomoco_dcm, nomoco_img = img_loader(nomoco_img_path)
moco_dcm, moco_img = img_loader(moco_img_path)

nomoco_amp, nomoco_amplog, nomoco_norm_amp = fourier(nomoco_img[60])
moco_amp, moco_amplog, moco_norm_amp = fourier(moco_img[60])

restored = ifourier(moco_amp)

diff_amp = moco_amp - nomoco_amp
restored_diff = ifourier(diff_amp)

plt.figure()
plt.imshow(restored_diff, cmap = 'jet')

xxx = moco_img[60] - nomoco_img[60]
xx_amp, xx_amplog, xx_norm_amp = fourier(xxx[60,:,:])

#######################################################
img = moco_amplog

histdraw(img, 300 ,[int(np.min(img)), int(np.max(img))])
#histdraw(img, 255 ,[0, 600])

########################################################



import math
import matplotlib



#curr_dir = os.getcwd()
#img = cv.imread(curr_dir+'/temp.png',0)

img = np.array(moco_img[60])

print( img.shape )

# Fourier Transform along the first axis

# Round up the size along this axis to an even number
n = int( math.ceil(img.shape[0] / 2.) * 2 )

# We use rfft since we are processing real values
a = np.fft.rfft(img,n, axis=0)

# Sum power along the second axis
a = a.real*a.real + a.imag*a.imag
a = a.sum(axis=1)/a.shape[1]

# Generate a list of frequencies
f = np.fft.rfftfreq(n)

# Graph it
plt.plot(f[1:],a[1:], label = 'sum of amplitudes over y vs f_x')

# Fourier Transform along the second axis

# Same steps as above
n = int( math.ceil(img.shape[1] / 2.) * 2 )

a = np.fft.rfft(img,n,axis=1)

a = a.real*a.real + a.imag*a.imag
a = a.sum(axis=0)/a.shape[0]

f = np.fft.rfftfreq(n)

plt.plot(f[1:],a[1:],  label ='sum of amplitudes over x vs f_y')

plt.ylabel( 'amplitude' )
plt.xlabel( 'frequency' )
plt.yscale( 'log' )

plt.legend()

plt.savefig( 'test_rfft.png' )
#plt.show()