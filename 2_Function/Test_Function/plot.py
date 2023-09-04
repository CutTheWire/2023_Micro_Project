import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm

# List of contour areas
contour_areas = [164.5, 165.5, 171.0, 172.5, 177.5, 185.0, 189.0, 189.0, 191.5, 194.0,
                 195.5, 196.5, 198.5, 199.5, 201.5, 202.0, 203.5, 205.0, 205.5, 207.0,
                 209.5, 210.5, 212.0, 212.5, 215.5, 217.5, 217.5, 218.0, 218.5, 222.5,
                 225.5, 225.5, 227.0, 227.5, 230.0, 232.0, 235.0, 236.5, 237.5, 238.5,
                 239.0, 240.0, 241.5, 242.5, 246.0, 252.0, 252.0, 252.5, 253.5, 257.5,
                 338.5, 362.0, 368.5, 426.0, 427.0, 441.0, 452.0, 465.0, 477.5, 481.0,
                 513.0, 779.0, 824.5, 1071.5, 1173.0, 1762.0]

# Fit normal distribution parameters
mu, std = norm.fit(contour_areas)

# Create subplots using GridSpec
fig = plt.figure(figsize=(8, 4))
grid = plt.GridSpec(1, 1, wspace=0.1, hspace=0.1)

# Plot normal distribution graph
ax = plt.subplot(grid[0, 0])

# Create a range of x values for the curve
x_range = np.linspace(min(contour_areas), max(contour_areas), 100)

# Plot histogram of contour areas
ax.hist(contour_areas, bins=20, density=True, alpha=0.6, color='c', label='Histogram')
# Plot normal distribution curve
ax.plot(x_range, norm.pdf(x_range, mu, std), 'r', label='Normal Distribution')
ax.set_title("Contour Area Distribution (Normal Distribution)")
ax.set_xlabel("Contour Area")
ax.set_ylabel("Density")
ax.legend()

plt.tight_layout()
plt.show()
