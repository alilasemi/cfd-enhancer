import os
import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd
import cv2



def load_images(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            img = io.imread(image_path)
            img = cv2.resize(img, (1152, 1758))
            images.append(color.rgb2gray(img).flatten())  # Convert to grayscale
    return np.array(images).T

def svd_projection(images, V):
    svd_result = V.T @ (images)
    
    return svd_result

def plot_images_in_space(image_projections, marker):
    markers = ['x', 's', 'o']
    labels = ['Fine', 'Coarse', 'SD']
    plt.scatter(image_projections[0, :], image_projections[1, :], marker=markers[marker], cmap='viridis', label=labels[marker])
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend()
    
def resize_images(images, target_size):
    resized_images = []
    for img in images:
        resized_img = cv2.resize(img, target_size)
        resized_images.append(resized_img)
    return resized_images

if __name__ == "__main__":
    # Directory paths for two sets of images
    set1_directory = "cylinder/fine/vorticity/"
    set2_directory = "cylinder/coarse/vorticity/"
    set3_directory = "output/"

    # Load images from both sets
    set1_images = load_images(set1_directory)
    set2_images = load_images(set2_directory)
    set3_images = load_images(set3_directory)
    
    V, S, Vt = randomized_svd(set1_images, n_components=3)

    set1_projection = svd_projection(set1_images, V)
    set2_projection = svd_projection(set2_images, V)
    set3_projection = svd_projection(set3_images, V)

    # Plot the images in space for both sets
    plot_images_in_space(set1_projection, 1)
    plot_images_in_space(set2_projection, 0)
    plot_images_in_space(set3_projection, 2)
    plt.show()
