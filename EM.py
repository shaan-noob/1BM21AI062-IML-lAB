import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
observed_data = np.concatenate([np.random.normal(-5, 1, 300), np.random.normal(5, 1, 300)]).reshape(-1, 1)

# Initialize the GMM with 2 components
gmm = GaussianMixture(n_components=2, random_state=42)

# Fit the GMM using the EM algorithm
gmm.fit(observed_data)

# Get the parameters
means = gmm.means_
covariances = gmm.covariances_
weights = gmm.weights_

# Predict the hidden variables (latent variable) using the EM algorithm
latent_variable = gmm.predict(observed_data)

# Plot the observed data and the estimated GMM distribution
x = np.linspace(-10, 10, 1000).reshape(-1, 1)
pdf_estimated = np.exp(gmm.score_samples(x))

plt.figure(figsize=(10, 6))

# Plot the observed data
plt.hist(observed_data, bins=30, density=True, alpha=0.5, label='Observed Data')

# Plot the estimated GMM distribution
plt.plot(x, pdf_estimated, 'g-', label='Estimated Distribution (GMM)')

plt.title('EM Algorithm for Gaussian Mixture Model')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
plt.show()
    
    
    
