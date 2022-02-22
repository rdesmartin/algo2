# Kmeans++

[source](http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf)

2. Randomly select the first centroid from the data points.
3. For each data point compute its distance from the nearest, previously chosen centroid.
4. Select the next centroid from the data points such that the probability of choosing a point as centroid is directly proportional to its distance from the nearest, previously chosen centroid. (i.e. the point having maximum distance from the nearest centroid is most likely to be selected next as a centroid)
5. Repeat steps 2 and 3 until k centroids have been sampled