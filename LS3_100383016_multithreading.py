import os
import numpy as np
from skimage import io, filters, color
from sklearn.metrics import jaccard_score
import threading
from threading import Semaphore
import time
import multiprocessing

sem = Semaphore(1)

def preprocess(image):
    image_preprocessed = filters.difference_of_gaussians(image, low_sigma = 120)
    return image_preprocessed

def segment(image):
    otsu_th = filters.threshold_otsu(image)
    image_segmented = (image < otsu_th).astype('int')
    return image_segmented

def postprocess(image):
    predicted_mask = image
    return predicted_mask


def skin_lesion_segmentation(img_root):
    image = io.imread(img_root)
    image_gray = color.rgb2gray(image)
    image_gray_preprocessed = preprocess(image_gray)
    image_segmented = segment(image_gray_preprocessed)
    predicted_mask = postprocess(image_segmented)
    return predicted_mask

def thread(img_roots,gt_mask_roots):
    predicted_mask = skin_lesion_segmentation(img_roots)
    gt_mask = io.imread(gt_mask_roots)/255 
    #sem.acquire()   
    #score.append(jaccard_score(np.ndarray.flatten(gt_mask),np.ndarray.flatten(predicted_mask)))
    #sem.release()
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
    NTHREADS = np.size(img_roots)

    pool_obj = multiprocessing.Pool()
    score = pool_obj.starmap(thread, zip(img_roots,gt_masks_roots))
    #t = threading.Thread(target=thread, args=[img_roots, gt_masks_roots])
    #t.start()
    #for i in np.arange(np.size(img_roots)):
        #t = threading.Thread(target=thread, args=[img_roots[i], gt_masks_roots[i]])
        #t.start()
        #print('I%d' %i)
        #score.append(t.join)

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
start = time.time()
mean_score = evaluate_masks(train_imgs_files, train_masks_files)
done = time.time()
print(done - start)
