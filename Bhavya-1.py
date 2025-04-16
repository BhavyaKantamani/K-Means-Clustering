import numpy as np
import matplotlib.pyplot as plt
import argparse


class KMeans:
    
    def __init__(self, n_clusters=3, init='k-means++', max_iter=100, tol=1e-6, random_state=0,n_init=None):
        
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_init = n_init
        self.centroids = None
        self.error = None
        
        np.random.seed(self.random_state)
        
        

    def initialize_centroids(self, X):

        
        if self.init == 'random':
            indices = np.random.choice(len(X), self.n_clusters, replace=False)
            centroids = [X[i] for i in indices]

        elif self.init == 'k-means++':
            centroids = [X[np.random.randint(len(X))]]
            while len(centroids) < self.n_clusters:
                
                distances = np.array([min([np.linalg.norm(x - c) for c in centroids]) for x in X])
                probabilities = distances / distances.sum()
                cumulative_probabilities = probabilities.cumsum()
                r = np.random.rand()
                for j, p in enumerate(cumulative_probabilities):
                    if r < p:
                        centroids.append(X[j])
                        break
        else:
            raise ValueError("Invalid value for 'init'. Use 'k-means++' or 'random'.")
            
        self.centroids = np.array(centroids)


    def fit(self, X):

        if self.n_init is None:
            self.n_init =1

        lowest_error = float('inf')

        for _ in range(self.n_init):
            self.initialize_centroids(X)
            for _ in range(self.max_iter):
                # Assign each data point to its nearest centroid
                clusters = [[] for _ in range(self.n_clusters)]
                distances_sum = 0
                for x in X:
                    distances = [np.linalg.norm(x - c) for c in self.centroids]
                    cluster_idx = np.argmin(distances)
                    distances_sum += min(distances)
                    clusters[cluster_idx].append(x)
                clusters = [np.array(cluster) for cluster in clusters]

                # Update centroids
                new_centroids = np.array([cluster.mean(axis=0) for cluster in clusters])

                #Check convergence
                if np.all(np.abs(self.centroids - new_centroids) < self.tol):
                    break 

                self.centroids = new_centroids

            # Update lowest error
            if distances_sum < lowest_error:
                lowest_error = distances_sum
                best_centroids = self.centroids
                best_clusters = clusters

        self.centroids = best_centroids
        self.error = lowest_error


        return

def read_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            row = list(map(float, line.split()))
            data.append(row)
    return np.array(data)




def k_means_plotter(file_path):
    data=read_data(file_path)
    final_data = data[:, :-1] # removing last column from the dataset
    k_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    errors=[] # Array to store sum of errors for all k values
    for i in k_values:
        k = KMeans(n_clusters=i,init = 'k-means++', max_iter = 20,random_state=0)  # Initialize a KMeans model with k clusters
        k.fit(final_data) # Fit the KMeans model to the data
        curr_error = k.error
        errors.append(curr_error)
        print(f"For K={i} After 20 iterations: Error = {round(curr_error,4)}")   # Print the Error for the current K value
    # plot for Error vs K
    plt.plot(k_values,errors)
    plt.xlabel("No.of Clusters - k")
    plt.ylabel("Error")
    plt.title(" Error vs k ")
    plt.show()
    


def main():
    
    parser = argparse.ArgumentParser(description='Implement KMeans')
    
    parser.add_argument('--file_path', type=str,default=None, help='Data file path')
    
    args = parser.parse_args()
    
    # Check if file_path is provided
    if not args.file_path:
        print("Please run the cmd in the format: KMeans.py --file_path <file_path>")
        return
    
    k_means_plotter(args.file_path)



    
    
if __name__=="__main__":
    
    main()
    
    
    
    