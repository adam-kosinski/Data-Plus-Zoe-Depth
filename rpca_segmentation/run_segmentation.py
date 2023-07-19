import numpy as np
import argparse
from skimage.feature import local_binary_pattern
from skimage import data
from PIL import Image
import os
from r_pca import R_pca


# local binary pattern (LBP) reference
# https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_local_binary_pattern.html

# morphological filter reference
# https://scikit-image.org/docs/stable/auto_examples/applications/plot_morphology.html


def show_numpy(np_img):
    np_img = np_img / np.max(np_img)
    Image.fromarray(np_img * (2**8-1)).show()

def save_numpy(np_img, fpath):
    np_img = np_img / np.max(np_img)
    Image.fromarray(np_img * (2**8-1)).convert("L").save(fpath)


def main(input_dir):
    # LBP config
    radius = 3
    n_points = 8*radius
    METHOD = "uniform"

    i_vectors = []
    filenames = []

    for file in os.listdir(input_dir):
        fpath = os.path.join(input_dir, file)
        ext = os.path.splitext(file)[1].lower()
        if not(ext == ".png" or ext == ".jpg" or ext == ".jpeg") or os.path.isdir(file):
            continue

        print(file)
        filenames.append(file)

        with Image.open(fpath).convert("L") as img:
            gray_img = np.asarray(img)
        
        lbp_img = local_binary_pattern(gray_img, n_points, radius, METHOD)

        i_star = 0.5*lbp_img + 0.5*gray_img
        i_vector = np.reshape(i_star, (i_star.shape[0]*i_star.shape[1], 1))
        i_vectors.append(i_vector)
        
    M = np.hstack(i_vectors)
    print(M.shape)

    rpca = R_pca(M)
    L, S = rpca.fit(max_iter=5, iter_print=1)

    print(S.shape)

    save_dir = os.path.join(input_dir, "sparse")
    os.makedirs(save_dir, exist_ok=True)

    for i in range(S.shape[1]):
        sparse_img = np.reshape(S[:,i], gray_img.shape)
        save_path = os.path.join(save_dir, os.path.splitext(filenames[i])[0] + ".png")
        save_numpy(sparse_img, save_path)
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    args = parser.parse_args()
    main(os.path.abspath(args.input_dir))