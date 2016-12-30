import numpy as np
import neuralnet, multilogit, kmeans, linear

algo_dict = {
    linear: 'linear',
    neuralnet: 'neuralnet',
    multilogit: 'multilogit',
    kmeans: 'kmeans'
}

for algo, algo_name in algo_dict.items():
    print(algo_name)
    errors = algo.start()
    np.savetxt("./statistical_analysis/"+algo_name+".csv", errors, delimiter=",")