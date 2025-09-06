import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from yellowbrick.cluster import SilhouetteVisualizer


dataset = pd.read_csv("dataset_preprocessed.csv")

X = dataset


distortions = []
inertias = [] # It is the sum of squared distances of samples to their closest cluster center.
mapping1 = {}
mapping2 = {}
K = range(2, 10)

plt.figure(num =1)
fig, ax = plt.subplots(4, 2, figsize=(15,8))  
for k in K:
    # Building and fitting the model
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
  
    inertias.append(kmeanModel.inertia_)
    
    mapping2[k] = kmeanModel.inertia_
    
    # Silhouette plot
    q, mod = divmod(k, 2)
    visualizer = SilhouetteVisualizer(kmeanModel, colors='yellowbrick', ax=ax[q-1][mod])
    visualizer.fit(X) 
    

plt.show()

plt.figure(num =2)
plt.plot(K, inertias, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Sum of squared distances')
plt.title('The Elbow Method for optimal K')
plt.show()
