import os
import numpy as np
from skimage import io, filters, color, morphology
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt
import multiprocessing


def preprocess(image):
    # First pre-processing technqiue: difference_of_gaussians
    image_preprocessed = filters.difference_of_gaussians(image, low_sigma = 140)
    # Second pre-processing technqiue: median filter
    image_preprocessed = filters.median(image_preprocessed)
    '''plt.figure()
    plt.subplot(1,2,1), plt.imshow(image)
    plt.subplot(1,2,2), plt.imshow(image_preprocessed)
    plt.show()'''
    return image_preprocessed

def segment(image):
    # Perform OTSU segmentation:
    # First we calculate the threshold
    otsu_th = filters.threshold_otsu(image)
    # After that, we get the values lower than that threshold
    image_segmented = (image < otsu_th).astype('int')
    return image_segmented

def postprocess(image):
    # First postprocess technqiue in order to isolate object from the rest: opening
    #predicted_mask = morphology.binary_opening(image) ESTA EN VERDAD NO HACE MUCHO
    # Second postprocess technqiue in order to fill the holes in the objects: dilation
    predicted_mask = morphology.binary_dilation(image)
    '''plt.figure()
    plt.subplot(1,2,1), plt.imshow(image)
    plt.subplot(1,2,2), plt.imshow(predicted_mask)
    plt.show()'''
    return predicted_mask


def skin_lesion_segmentation(img_root):
    # Read the image
    image = io.imread(img_root)
    # Cinvert the incoming image into a gray image
    image_gray = color.rgb2gray(image)
    # Preprocess the image
    image_gray_preprocessed = preprocess(image_gray)
    # Segmenetation process
    image_segmented = segment(image_gray_preprocessed)
    # Postprocessing of the image
    predicted_mask = postprocess(image_segmented)
    return predicted_mask

def thread(img_roots,gt_mask_roots):
    # Perform the segmentation with all its preprocessing and postprocessing
    predicted_mask = skin_lesion_segmentation(img_roots)
    gt_mask = io.imread(gt_mask_roots)/255 
    # Get the jaccard score compraing the result with the mask of the set
    return jaccard_score(np.ndarray.flatten(gt_mask),np.ndarray.flatten(predicted_mask))

def evaluate_masks(img_roots, gt_masks_roots):
    """ EVALUATE_MASKS: 
        It receives two lists:
        1) A list of the file names of the images to be analysed
        2) A list of the file names of the corresponding Ground-Truth (GT) 
            segmentations
        For each image on the list:
            It performs the segmentation
            It determines Jaccard Index
        Finally, it computes an average Jaccard Index for all the images in 
        the list
    """
    score = []
    # We create a pool of mupltiprocess in order to perfrom a faster training
    pool_obj = multiprocessing.Pool()
    # Get the score of each image after training
    score = pool_obj.starmap(thread, zip(img_roots,gt_masks_roots))
    # Calculate the mean of the image scores
    mean_score = np.mean(score)
    print('Average Jaccard Index: '+str(mean_score))
    return mean_score

# -----------------------------------------------------------------------------
#
#     READING IMAGES
#
# -----------------------------------------------------------------------------

data_dir= os.curdir

train_imgs_files = [ os.path.join(data_dir,'train50/images',f) for f in sorted(os.listdir(os.path.join(data_dir,'train50/images'))) 
            if (os.path.isfile(os.path.join(data_dir,'train50/images',f)) and f.endswith('.jpg')) ]

train_masks_files = [ os.path.join(data_dir,'train50/masks',f) for f in sorted(os.listdir(os.path.join(data_dir,'train50/masks'))) 
            if (os.path.isfile(os.path.join(data_dir,'train50/masks',f)) and f.endswith('.png')) ]

# train_imgs_files.sort()
# train_masks_files.sort()
print("Number of train images", len(train_imgs_files))
print("Number of image masks", len(train_masks_files))

# -----------------------------------------------------------------------------
#
#     Segmentation and evaluation
#
# -----------------------------------------------------------------------------
mean_score = evaluate_masks(train_imgs_files, train_masks_files)
