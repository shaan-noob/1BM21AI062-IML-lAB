import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Standardize the features (important for PCA)
X_std = StandardScaler().fit_transform(X)

# Apply PCA with 2 principal components
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_std)

# Create a DataFrame for visualization
pc_df = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])
pc_df['Target'] = y

# Plot the 2D projection of the data
plt.figure(figsize=(8, 6))
targets = list(iris.target_names)
colors = ['r', 'g', 'b']

for target, color in zip(targets, colors):
    indices_to_keep = pc_df['Target'] == list(iris.target_names).index(target)
    plt.scatter(pc_df.loc[indices_to_keep, 'Principal Component 1'],
                pc_df.loc[indices_to_keep, 'Principal Component 2'],
                c=color, s=50)

plt.title('2D Projection of Iris Dataset using PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(targets)
plt.grid(True)
plt.show()
