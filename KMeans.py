import numpy as np
import io
from scipy.spatial import distance as SCD
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import cm
#--------------------------------Initial Values-------------------------
filename='new_dataset_1.txt'
k=int(input("Please enter number of clusters: "))
initial=[int(x) for x in input("Please enter Initial Centroid IDs: ").split()]
iterations=int(input("Please enter Number of Iterations: "))
for i in range(len(initial)):
    initial[i]=initial[i]-1
#---------------------Fetching Number of Columns------------------------
first_line = []
with open(filename, 'r') as f:
    first_line = f.readline()
n = len(first_line.split("	"))
#-------------------To read as a matrix---------------------------------
input_file = np.loadtxt(filename,usecols=range(2,n))
f=open(filename,"r")
lines=f.readlines()
Ground_truth=[]
for x in lines:
    Ground_truth.append(x.split('\t')[1])
f.close()
#------------------------Initial values of G-----------------------------
G=[]
for x in initial:
    G.append(input_file[x])
#-----------------------------KMeans-------------------------------------

clusters=np.empty([len(input_file)])
for p in range(iterations):    
    for i in range(len(input_file)):
        dist=[]
        for j in G:
            dist.append(SCD.euclidean(input_file[i],j))
        clusters[i]=(np.argmin(dist))
    Gnew=[]
    for m in range (0,k):
        temp=[i for i, x in enumerate(clusters) if x == m]
        Gnew.append(np.mean(input_file[temp,:],axis=0))
    Gnew=np.asarray(Gnew)
    G=np.copy(Gnew)
#-----------------------Calculation of Jaccard and Rand--------------------
GroundTruthMatrix=np.empty([len(input_file),len(input_file)])
for i in range(len(input_file)):
    for j in range(len(input_file)):
        if Ground_truth[i]==Ground_truth[j]:
            GroundTruthMatrix[i][j]=1
        else:
            GroundTruthMatrix[i][j]=0
            
ClusteringMatrix=np.empty([len(input_file),len(input_file)])
for i in range(len(input_file)):
    for j in range(len(input_file)):
        if clusters[i]==clusters[j]:
            ClusteringMatrix[i][j]=1
        else:
            ClusteringMatrix[i][j]=0
m11=0
m10=0
m00=0
for i in range(len(input_file)):
    for j in range(len(input_file)):
        if ClusteringMatrix[i][j]==GroundTruthMatrix[i][j]:
            if ClusteringMatrix[i][j]==1:
                m11=m11+1
            else:
                m00=m00+1
        else:
            m10=m10+1
Jaccard=m11/(m11+m10)
print("Jaccard is ",Jaccard)
Rand=(m11+m00)/(m11+m00+m10)
print("Rand is ",Rand)
#-----------------------PCA conversion-----------
pca = PCA(n_components=2)
yi=pca.fit_transform(input_file)
xcoord=yi[:,(0)]
ycoord=yi[:,(1)]
#-----------------------Plotting------------------
for j in range(k):
    indices = [i for i, x in enumerate(clusters) if x == j]
    plt.scatter(xcoord[indices],ycoord[indices],cmap=cm.get_cmap('Dark2'),label=j+1)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title("Kmeans Clustering on "+filename+" Using Centroids 5,25,32,100,132")
plt.show()
#------------------Printing Clusters----------------
for j in range(k):
    print("Cluster ",j+1)
    print([i for i, x in enumerate(clusters) if x == j])