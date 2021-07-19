import folium 
import pandas as pd 
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
import Working_with_the_algorithms
import sklearn.cluster as sk
import numpy as np

df = pd.read_csv('listings.csv')

def generateBaseMap(default_location = [40.693943,-73.985880]):
    base_map = folium.Map(location=default_location)
    return base_map 

base_map = generateBaseMap()

HeatMap(data=df[['latitude','longitude','price']].groupby(['latitude','longitude']).mean().reset_index().values.tolist(), radius=8, max_zoom=13).add_to(base_map)

base_map.save('index.html')

"""
It is not very useful to draw conclusions on prices in new york city - the map seems generally uniform
in terms of its color, so it is not easy to see where has high and low prices.
"""

def separateintoclusters(X, clusters):
    numcentroids = np.amax(clusters) + 1
    centroids = [[]] * numcentroids

    for i in range(len(clusters)):
        cluster = clusters[i]
        point = X[i]
        centroids[cluster] = centroids[cluster] + [point]
    return np.array(centroids)

def plot_points(X, clusteralgo):
    organizedpoints = separateintoclusters(X, clusteralgo(X))
    fig=plt.figure()
    ax=fig.add_axes([0,0,1,1])
    colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
    for i in range(len(organizedpoints)):
        clusterdata = np.array(organizedpoints[i])
        latitudedata = clusterdata[:, 0]
        longitutedata = clusterdata[:, 1]
        ax.scatter(latitudedata, longitutedata, color=colors[i])
    plt.show()

def avgprice(X, clusteralgo):
    organizedpoints = separateintoclusters(X, clusteralgo(X))
    for i in range(len(organizedpoints)):
        clusterdata = np.array(organizedpoints[i])
        prices = clusterdata[:, 2]
        print('cluster', i, 'average price:', np.average(prices))

"""
Yes, this is in general what I had in mind about NYC. It seems as though the bureaus have been grouped up,
and I know that each of the bureaus have their own living situation and varying expensiveness. So,
the beauraus all being in their own cluster makes sense to me.
"""

data = Working_with_the_algorithms.import_data('listings.csv')

print('First, k-means...')

plot_points(data, Working_with_the_algorithms.kmeans)
avgprice(data, Working_with_the_algorithms.kmeans)

""" 
Note: I have commented this part out for your convinience as it takes approximately an hour to run

print('Next, hierarchical clustering...')
plot_points(data, Working_with_the_algorithms.hierarchical)
avgprice(data, Working_with_the_algorithms.hierarchical) """


print('Finally, GMM...')

plot_points(data, Working_with_the_algorithms.gmm)
avgprice(data, Working_with_the_algorithms.gmm)