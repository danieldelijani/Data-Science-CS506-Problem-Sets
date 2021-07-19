import sklearn.cluster as sk
import sklearn.mixture as mix
import pandas as pd
import numpy as np

def import_data(filename): 

    df = pd.read_csv(filename)
    data = df[['longitude', 'latitude', 'price']]
    npdata = data.to_numpy()

    r = 0

    for ignoredvar in range(len(npdata)):
        for c in range(len(npdata[0])):
            if npdata[r][c] == '':
                 npdata = np.delete(npdata, r)
                 r -= 1
        r += 1
            
    return npdata

def kmeans(data):
    model = sk.KMeans(n_clusters=5 ,init="k-means++").fit_predict(data)
    return model

def hierarchical(data):
    # n_clusters=2, affinity= 'euclidean', compute_full_tree=False
    model = sk.AgglomerativeClustering(n_clusters=3, linkage='average').fit_predict(data)
    return model

def gmm(data):
    model = mix.GaussianMixture(n_components=5).fit_predict(data)
    return model

"""
Parameter explanation:

kmeans:
n_clusters = 5 because I assume there will be clusterings for the 5 bureaus of New York

hierarchical:
n_clusters = 3 because even with 3 clusters this algorithm takes an incredibly long time to run
linkage = 'average' because I think it is the most effective when also considering time complexity, which I have found to be an issue with this function

gmm:
n_components=5 because once again I assume there will be clustering amont the 5 bereaus of New York

"""


"""
2b)
K-means:

Pros:
Generalizes well to clusters of different shapes and sizes
Scales well to large datasets
Cons:
Poorly clusters data of varying density
Outliers disproportionally affect model

Hierarchical:

Pros:
Easy to implement
Easier to decipher how many clusters is optimal based on dendrogram

Cons:
Very sensitive to outliers
Time complexity not good for large datasets

GMM:

Pros:
Strongly consistent with large sets of data
GMM is a lot more flexible in terms of cluster covariance

Cons:
Model can have large bias and inefficiency in small samples
Sensitive to normalizations of the model or parameters
"""