import timeit
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA as ICA
from sklearn.decomposition import TruncatedSVD as SVD
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score as sil_score, f1_score, homogeneity_score

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

df = pd.read_csv('heart_failure_clinical_records_dataset.csv',header=None)

x = df.iloc[:,0:-1]
y = df.iloc[:,-1]

sc = StandardScaler()
x = sc.fit_transform(x)

test_km_acc = {}
train_km_acc = {}
test_em_acc = {}
train_em_acc = {}
km_time = {}
em_time = {}

x_train, x_test, y_train, y_test = train_test_split(x,y ,test_size=0.2,random_state=0)

def kmeans_alg(x,y):
    km_sse = {}
    km_sil = {}
    km_f1 = {}
    km_homo = {}
    for i in range(2,11,2):
        km = KMeans(n_clusters=i,random_state=10).fit(x)
        km_sil[i] = sil_score(x,km.predict(x),metric='euclidean')
        km_f1[i] = f1_score(y,km.predict(x),average='micro')
        km_homo[i] = homogeneity_score(y,km.predict(x))
        km_sse[i] = km.inertia_
    
    plt.plot(list(km_sse.keys()),list(km_sse.values()))
    plt.xlabel("Number of cluster")
    plt.ylabel("SSE")
    plt.title('No.of Clusters Vs SSE using Kmeans')
    plt.show()

    plt.plot(list(km_sil.keys()),list(km_sil.values()))
    plt.xlabel("Number of cluster")
    plt.ylabel("Silhoutte Coefficient")
    plt.title('No.of Clusters Vs Silhoutte Coefficient using Kmeans')
    plt.show()

    plt.plot(list(km_f1.keys()),list(km_f1.values()))
    plt.xlabel("Number of cluster")
    plt.ylabel("F1 Scores")
    plt.title('No.of Clusters Vs F1-scores using Kmeans')
    plt.show()

    plt.plot(list(km_homo.keys()),list(km_homo.values()))
    plt.xlabel("Number of cluster")
    plt.ylabel("Homogeneity score")
    plt.title('No.of Clusters Vs Homogenity score using Kmeans')
    plt.show()
    
kmeans_alg(x,y)
kmeans = KMeans(n_clusters=4, random_state=0).fit(x)
# Visualize clusters Parallel plot
def kmeans_parallel_plot(kmeans_obj, input_x_pd, n):
    df1 = df
    df1['class'] = kmeans_obj.labels_
    rand_idx1 = list(range(0,n))
    idx_viz1 = np.append(rand_idx1, [df1.shape[1] - 2,df1.shape[1] - 1])
    plt.figure(figsize=(8, 6), dpi=200)
    pd.plotting.parallel_coordinates(df1.iloc[:, idx_viz1], 'class', colormap='Set1')
    plt.xlabel('Features')
    plt.ylabel('Value')
    plt.title('k-means visualization')
    plt.show()


kmeans_parallel_plot(kmeans, x,6)

def em_alg(x,y):
    em_sil = {}
    em_f1 = {}
    em_homo  = {}
    em_aic = {}
    em_bic = {}
    for k in range(2,11,2):
        em = GaussianMixture(n_components=k,covariance_type='diag',n_init=1,warm_start=True,random_state=100).fit(x)
        label = em.predict(x)
        em_sil[k] = sil_score(x, label, metric='euclidean')
        em_f1[k] = f1_score(y, em.predict(x), average='micro')
        em_homo[k] = homogeneity_score(y, em.predict(x))
        em_bic[k] = em.bic(x)
        em_aic[k] = em.aic(x)
    
    plt.plot(list(em_sil.keys()),list(em_sil.values()))
    plt.xlabel("Number of cluster")
    plt.ylabel("Silhoutte Coefficient")
    plt.title('No.of Clusters Vs Silhoutte Coefficient using EM')
    plt.show()

    plt.plot(list(em_f1.keys()),list(em_f1.values()))
    plt.xlabel("Number of cluster")
    plt.ylabel("F1 Scores")
    plt.title('No.of Clusters Vs F1-scores using EM')
    plt.show()

    plt.plot(list(em_homo.keys()),list(em_homo.values()))
    plt.xlabel("Number of cluster")
    plt.ylabel("Homogeneity score")
    plt.title('No.of Clusters Vs Homogenity score using EM')
    plt.show()

    plt.plot(list(em_bic.keys()), list(em_bic.values()))
    plt.xlabel("Number of cluster")
    plt.ylabel("BIC")
    plt.title('No.of Clusters Vs Bayesian Information Criterion using EM')
    plt.show()

    plt.plot(list(em_aic.keys()),list(em_aic.values()))
    plt.xlabel("Number of cluster")
    plt.ylabel("AIC")
    plt.title('No.of Clusters Vs Akaike Information Criterion using EM')
    plt.show()
    
em_alg(x,y)
em = GaussianMixture(n_components=5,random_state=100).fit(x)

def em_parallel_plot(input_x_pd, n,labels):
    df1 = df
    df1['class'] = labels
    rand_idx1 = list(range(0,n))
    idx_viz1 = np.append(rand_idx1, [df1.shape[1] - 2,df1.shape[1] - 1])
    plt.figure(figsize=(8, 6), dpi=200)
    pd.plotting.parallel_coordinates(df1.iloc[:, idx_viz1], 'class', colormap='Set1')
    plt.xlabel('Features')
    plt.ylabel('Value')
    plt.title('Em visualization')
    plt.tight_layout()
    plt.show()

em_parallel_plot(x,6,em.predict(x))
#Kmeans
km = KMeans(n_clusters=5,random_state=10)
km_x_train = km.fit_transform(x_train)
km_x_test = km.fit_transform(x_test)

#PCA
pca = PCA(n_components=1)
pca_km_train = pca.fit_transform(km_x_train)
pca_km_test = pca.fit_transform(km_x_test)
print('"""""""""" PCA Kmeans & EM plots """"""""""')
pca_em = pca.fit_transform(x)
pca.fit(x,y)
kmeans_alg(x,y)
em_alg(x,y)
kmeans_parallel_plot(kmeans, x,6)
em_parallel_plot(x,6,em.predict(x))

#ICA
ica = ICA(n_components=5)
ica_km_train = ica.fit_transform(km_x_train)
ica_km_test = ica.fit_transform(km_x_test)
print('"""""""""" ICA Kmeans & EM plots """"""""""')
ica_em = ica.fit_transform(x)
ica.fit(x,y)
kmeans_alg(x,y)
em_alg(x,y)
kmeans_parallel_plot(kmeans, x,6)
em_parallel_plot(x,6,em.predict(x))

#SVD
svd = SVD(n_components=2)
svd_km_train = svd.fit_transform(km_x_train)
svd_km_test = svd.fit_transform(km_x_test)
print('"""""""""" SVD Kmeans & EM plots """"""""""')
svd_em = svd.fit_transform(x)
svd.fit(x,y)
kmeans_alg(x,y)
em_alg(x,y)
kmeans_parallel_plot(kmeans, x,6)
em_parallel_plot(x,6,em.predict(x))

#Gaussian Random Projection[GRP]
grp = GaussianRandomProjection(n_components=5)
grp_km_train = grp.fit_transform(km_x_train)
grp_km_test = grp.fit_transform(km_x_test)
print('"""""""""" GRP Kmeans & EM plots """"""""""')
grp_em = grp.fit_transform(x)
grp.fit_transform(x,y)
kmeans_alg(x,y)
kmeans_parallel_plot(kmeans, x, 6)
em_parallel_plot(x,6,em.predict(x))


def train(x_train,y_train):
    adam = Adam(lr=0.0001)

    nn_classifier = Sequential()

    # Adding the input layer and the first hidden layer
    nn_classifier.add(Dense(units=50, activation = 'relu'))

    # Adding the second hidden layer
    nn_classifier.add(Dense(units=50, activation = 'relu'))

    # Adding the output layer
    nn_classifier.add(Dense(units=8,activation = 'softmax'))

    # Compiling the ANN
    nn_classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

    # Fitting the ANN to the Training set
    history = nn_classifier.fit(x_train, y_train, batch_size = 4, epochs = 10, validation_split=0.1)
    
    return nn_classifier
# PCA on Kmeans
print('""""""" Traininig PCA kmeans data using NN """""""')
start = timeit.default_timer()
pca_km_nn = train(pca_km_train,y_train)
km_time['PCA'] = timeit.default_timer() - start
y_pred = pca_km_nn.predict(pca_km_test)
test_km_acc['PCA'] = accuracy_score(y_test,np.argmax(y_pred,axis=1))
print('PCA Kmeans Confusion matrix : ',confusion_matrix(y_test,np.argmax(y_pred,axis=1)))
train_km_acc['PCA'] = accuracy_score(y_train,np.argmax(pca_km_nn.predict(pca_km_train),axis=1))

# PCA on Expectation Maximization(EM)
print('""""""" Traininig PCA EM data using NN """""""')
start = timeit.default_timer()
pca_em_nn = train(x_train,y_train)
em_time['PCA'] = timeit.default_timer() - start
y_pred = pca_em_nn.predict(x_test)
test_em_acc['PCA'] = accuracy_score(y_test,np.argmax(y_pred,axis=1))
print('PCA EM Confusion matrix : ',confusion_matrix(y_test,np.argmax(y_pred,axis=1)))
train_em_acc['PCA'] = accuracy_score(y_train,np.argmax(pca_em_nn.predict(x_train),axis=1))

# ICA on Kmeans
print('""""""" Traininig ICA kmeans data using NN """""""')
start = timeit.default_timer()
ica_km_nn = train(ica_km_train,y_train)
km_time['ICA'] = timeit.default_timer() - start
y_pred1 = ica_km_nn.predict(ica_km_test)
test_km_acc['ICA'] = accuracy_score(y_test,np.argmax(y_pred1,axis=1))
print('ICA Kmeans Confusion matrix : ',confusion_matrix(y_test,np.argmax(y_pred1,axis=1)))
train_km_acc['ICA'] = accuracy_score(y_train,np.argmax(ica_km_nn.predict(ica_km_train),axis=1))

# ICA on Expectation Maximization(EM)
print('""""""" Traininig ICA EM data using NN """""""')
start = timeit.default_timer()
ica_em_nn = train(x_train,y_train)
em_time['ICA'] = timeit.default_timer() - start
y_pred1 = ica_em_nn.predict(x_test)
test_em_acc['ICA'] = accuracy_score(y_test,np.argmax(y_pred1,axis=1))
print('ICA EM Confusion matrix : ',confusion_matrix(y_test,np.argmax(y_pred1,axis=1)))
train_em_acc['ICA'] = accuracy_score(y_train,np.argmax(ica_em_nn.predict(x_train),axis=1))

#SVD On Kmeans
print('""""""" Traininig SVD kmeans data using NN """""""')
start = timeit.default_timer()
svd_km_nn = train(svd_km_train,y_train)
km_time['SVD'] = timeit.default_timer() - start
y_pred2 = svd_km_nn.predict(svd_km_test)
test_km_acc['SVD'] = accuracy_score(y_test,np.argmax(y_pred2,axis=1))
print('SVD Kmeans Confusion matrix : ',confusion_matrix(y_test,np.argmax(y_pred2,axis=1)))
train_km_acc['SVD'] = accuracy_score(y_train,np.argmax(svd_km_nn.predict(svd_km_train),axis=1))

# SVD on Expectation Maximization(EM)
print('""""""" Traininig SVD EM data using NN """""""')
start = timeit.default_timer()
svd_em_nn = train(x_train,y_train)
em_time['SVD'] = timeit.default_timer() - start
y_pred2 = svd_em_nn.predict(x_test)
test_em_acc['SVD'] = accuracy_score(y_test,np.argmax(y_pred2,axis=1))
print('SVD EM Confusion matrix : ',confusion_matrix(y_test,np.argmax(y_pred2,axis=1)))
train_em_acc['SVD'] = accuracy_score(y_train,np.argmax(svd_em_nn.predict(x_train),axis=1))

# GRP on Kmeans
print('""""""" Traininig GRP kmeans data using NN """""""')
start = timeit.default_timer()
grp_km_nn = train(grp_km_train,y_train)
km_time['GRP'] = timeit.default_timer() - start
y_pred3 = grp_km_nn.predict(grp_km_test)
test_km_acc['GRP'] = accuracy_score(y_test,np.argmax(y_pred3,axis=1))
print('GRP Kmeans Confusion matrix : ',confusion_matrix(y_test,np.argmax(y_pred3,axis=1)))
train_km_acc['GRP'] = accuracy_score(y_train,np.argmax(grp_km_nn.predict(grp_km_train),axis=1))

# GRP on Expectation Maximization(EM)
print('""""""" Traininig GRP EM data using NN """""""')
start = timeit.default_timer()
grp_em_nn = train(x_train,y_train)
em_time['GRP'] = timeit.default_timer() - start
y_pred3 = grp_em_nn.predict(x_test)
test_em_acc['GRP'] = accuracy_score(y_test,np.argmax(y_pred3,axis=1))
print('GRP Kmeans Confusion matrix : ',confusion_matrix(y_test,np.argmax(y_pred3,axis=1)))
train_em_acc['GRP'] = accuracy_score(y_train,np.argmax(grp_em_nn.predict(x_train),axis=1))


km_test_acc_table = pd.DataFrame(test_km_acc,index=['test'])
km_train_acc_table = pd.DataFrame(train_km_acc,index=['train'])
km_time_table = pd.DataFrame(km_time,index=['time'])
km_table = pd.concat([km_test_acc_table,km_train_acc_table,km_time_table])
print('"""" Kmeans Train and Test accuracies after applying PCA ICA SVD """"')
print(km_table)
print('\n')

em_test_acc_table = pd.DataFrame(test_em_acc,index=['test'])
em_train_acc_table = pd.DataFrame(train_em_acc,index=['train'])
em_time_table = pd.DataFrame(em_time,index=['time'])
em_table = pd.concat([em_test_acc_table,em_train_acc_table,em_time_table])
print('"""" EM Train and Test accuracies after applying PCA ICA SVD """"')
print(em_table)
print('\n')

