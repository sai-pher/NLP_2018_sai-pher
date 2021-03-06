import nltk
from gensim.models import Word2Vec
from nltk.cluster import KMeansClusterer
from sklearn import cluster
from sklearn import metrics

# training data
sentences = [['this', 'is', 'the', 'one', 'good', 'machine', 'learning', 'book'],
             ['this', 'is', 'another', 'book'],
             ['one', 'more', 'book'],
             ['weather', 'rain', 'snow'],
             ['yesterday', 'weather', 'snow'],
             ['forecast', 'tomorrow', 'rain', 'snow'],
             ['this', 'is', 'the', 'new', 'post'],
             ['this', 'is', 'about', 'more', 'machine', 'learning', 'post'],
             ['and', 'this', 'is', 'the', 'one', 'last', 'post', 'book']]

# training model
model = Word2Vec(sentences, min_count=1, size=300)
# print(type(model))
# print(model.)

# get vector data
X = model.wv.vocab
print("printing X")
print(type(X))
print(model.similarity('this', 'is'))
print(model.similarity('post', 'book'))
print(model.most_similar(positive=['machine'], negative=[], topn=2))
# print(model['the'])
print('print block')
# print(list(model.wv.vocab))
# print(len(list(model.wv.vocab)))

NUM_CLUSTERS = 3
kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)

assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
print(assigned_clusters)

words = model.vocabulary
for i, word in enumerate(words):
    print(word + ":" + str(assigned_clusters[i]))

kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS)
kmeans.fit(X)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print("Cluster id labels for inputted data")
print(labels)
print("Centroids data")
print(centroids)

print(
    "Score (Opposite of the value of X on the K-means objective which is Sum of distances of samples to their closest cluster center):")
print(kmeans.score(X))

silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')

print("Silhouette_score: ")
print(silhouette_score)
