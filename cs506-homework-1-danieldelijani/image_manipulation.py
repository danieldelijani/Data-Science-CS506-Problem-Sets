import k_means_clustering
import cv2
import numpy as np
from PIL import Image
import matplotlib


def createsmallerimage(filename, scale):

    foo = Image.open('boston-1993606_1280.jpg')
    print(foo.size)

    size1 = int(foo.size[0] * scale)
    size2 = int(foo.size[1] * scale)

    foo = foo.resize((size1, size2))

    foo.save('compressedimg.jpeg', 'JPEG', dpi=[300,300], quality=100)

    img = cv2.imread('compressedimg.jpeg')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def converttotwodarray(X):
    """indices = np.dstack(np.indices(X.shape[:2]))
    data = np.concatenate((X, indices), axis=0)
    return data"""

    X = X.tolist()
    finalarray = []
    for r in range(len(X)):
        finalarray = finalarray + X[r]
    return np.array(finalarray)

def imgkmeans(img):

    reformatted = converttotwodarray(img)

    initial_centroids =  k_means_clustering.choose_random_centroids(reformatted, K=10)
    clusters, centroid_history = k_means_clustering.run_k_means(reformatted, initial_centroids, n_iter=10)
    centroids = centroid_history[-1]
    print(centroids)

    newimg = np.reshape(clusters, img.shape[0:2])

    newimg = newimg.tolist()

    for r in range(len(newimg)):
        for c in range(len(newimg[0])):
            cluster = newimg[r][c]
            val = centroids[cluster]
            newimg[r][c] = val
    newimg = np.array(newimg)
    return newimg



img = createsmallerimage('boston-1993606_1280.jpg', .5)
newimg = imgkmeans(img)
im = Image.fromarray(newimg.astype(np.uint8), mode='RGB')
im.save('final.jpeg')