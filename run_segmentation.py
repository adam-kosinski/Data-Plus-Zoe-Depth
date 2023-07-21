import numpy as np
import argparse
from skimage.feature import local_binary_pattern
from skimage.exposure import equalize_hist
from skimage import filters
from skimage.morphology import disk, opening
from PIL import Image
import os
from r_pca import R_pca
import random

# code is an implementation of this paper:
# https://link.springer.com/content/pdf/10.1007/s00371-017-1463-9.pdf
# thresholding is done below per-image, instead of to the whole sparse matrix as done in the paper

# local binary pattern (LBP) reference
# https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_local_binary_pattern.html

# morphological filter reference
# https://scikit-image.org/docs/stable/auto_examples/applications/plot_morphology.html

def show_numpy(np_image):
    if np.max(np_image > 0):
        np_image = np_image / np.max(np_image)
    Image.fromarray(np_image * (2**8-1)).show()


def save_numpy(np_image, fpath):
    if np.max(np_image > 0):
        np_image = np_image / np.max(np_image)
    Image.fromarray(np_image * (2**8-1)).convert("L").save(fpath)

def save_step(np_image, input_dir, step_name, file):
    save_dir = os.path.join(input_dir, step_name)
    os.makedirs(save_dir, exist_ok=True)
    save_numpy(np_image, os.path.join(save_dir, os.path.splitext(file)[0] + ".png"))


def is_night(pil_image):
    np_image = np.asarray(pil_image.convert("RGB"))
    for i in range(5):
        x = random.randint(0, np_image.shape[1]-1)
        y = random.randint(0, np_image.shape[0]-1)
        equal = np_image[y,x,0] == np_image[y,x,1] and np_image[y,x,1] == np_image[y,x,2]
        if not equal:
            return False
    return True


def type1_preprocess(pil_image, beta=0):
    print("type 1 preprocess")
    gray_image = np.asarray(pil_image.convert("L"))
    equalized_image = equalize_hist(gray_image) * 255
    equalized_image = equalized_image.astype(int)

    radius = 1
    n_points = 8*radius
    METHOD = "uniform"
    lbp_img = local_binary_pattern(equalized_image, n_points, radius, METHOD)

    i_star = beta*lbp_img + (1-beta)*equalized_image
    return i_star, lbp_img


def type2_preprocess(pil_image, beta=0.35):
    print("type 2 preprocess")
    gray_image = np.asarray(pil_image.convert("L"))
    blurred_image = filters.gaussian(gray_image, sigma=0.5, preserve_range=True).astype(int)

    radius = 2
    n_points = 8*radius
    METHOD = "uniform"
    lbp_img = local_binary_pattern(blurred_image, n_points, radius, METHOD)

    i_star = beta*lbp_img + (1-beta)*blurred_image
    return i_star, lbp_img



def main(input_dir, output_dir=None, resize_factor=4, save_steps=False, inference_files=None):
    # returns a dictionary: key = image absolute path, value = segmentation mask absolute path
    # if inference files is specified, will only save final output for those files
    out_dictionary = {}

    if not output_dir:
        output_dir = os.path.join(input_dir, "output")
    
    vectors = []
    filenames = []
    gray_image_shape = None

    # Load and preprocess images

    for file in os.listdir(input_dir):
        fpath = os.path.join(input_dir, file)
        ext = os.path.splitext(file)[1].lower()
        if os.path.isdir(file) or not(ext == ".png" or ext == ".jpg" or ext == ".jpeg"):
            continue

        print(file)
        filenames.append(file)

        with Image.open(fpath) as image:
            image = image.resize((image.width // resize_factor, image.height // resize_factor))

            preprocessed_image, lbp_image = type2_preprocess(image) if is_night(image) else type1_preprocess(image)

            if save_steps:
                save_step(lbp_image, input_dir, "LBP", file)
                save_step(preprocessed_image, input_dir, "preprocessed", file)
            
            gray_image_shape = preprocessed_image.shape # store so we know how to reshape back to an image later
            vector = np.reshape(preprocessed_image, (preprocessed_image.shape[0]*preprocessed_image.shape[1], 1))
            vectors.append(vector)
    

    # Prep data matrix and do RPCA

    M = np.hstack(vectors)
    print(M.shape)
    rpca = R_pca(M)
    L, S = rpca.fit(max_iter=5, iter_print=1)
    print("rpca done")


    # post processing and save

    median_footprint = np.ones((3,3))
    morphological_footprint = disk(1)
    os.makedirs(output_dir, exist_ok=True)

    for i in range(S.shape[1]):
        if inference_files and os.path.abspath(os.path.join(input_dir, filenames[i])) not in inference_files:
            continue

        save_filename = os.path.splitext(filenames[i])[0] + ".png"
        print(save_filename)

        sparse_img = np.reshape(S[:,i], gray_image_shape)

        threshold = np.std(sparse_img, ddof=1)
        thresholded = (np.multiply(sparse_img, sparse_img) > threshold) * 255

        filtered = filters.median(thresholded, median_footprint)
        filtered = opening(filtered, morphological_footprint)
        
        if save_steps:
            save_step(sparse_img, input_dir, "sparse", filenames[i])
            save_step(thresholded, input_dir, "sparse_thresholded", filenames[i])

        save_path = os.path.join(output_dir, save_filename)
        save_numpy(filtered, save_path)
    
        orig_abs_path = os.path.abspath(os.path.join(input_dir, filenames[i]))
        save_abs_path = os.path.abspath(save_path)
        out_dictionary[orig_abs_path] = save_abs_path
    
    return out_dictionary



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    args = parser.parse_args()
    main(os.path.abspath(args.input_dir), None, True)