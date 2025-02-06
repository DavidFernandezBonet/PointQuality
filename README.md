# PointQuality

**PointQuality** is a package for computing quality metrics to compare point clouds.


Especially useful when evaluating reconstructed point clouds where only the relative spatial relationships are preserved (as opposed to the affine transformations of translation, rotation, chirality, and scale). 


# Metrics
- **Local Metrics:**
  - **KNN Metric:** Compare the nearest neighbors between original and reconstructed point clouds.
  - **GTA KNN Metric:** Compute neighborhood preservation using a ground truth graph.
  - **Distortion Measurement:** Compute the median distortion between corresponding points.


- **Global Metrics (GTA):**
  - **CPD Metric:** Evaluate the correlation of pairwise distances (using Pearson correlation) between the two point clouds.
  - **GTA CPD Metric:** Compare the graph-based distances with distances in the reconstructed point cloud.

## Installation

Install via pip (if published on PyPI):

```bash
pip install PointQuality
```


## Requirements

- Python >= 3.6
- [NumPy](https://numpy.org/)
- [SciPy](https://www.scipy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [NetworkX](https://networkx.org/)
- [python-igraph](https://igraph.org/python/)

You can also install these dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

Below is a quick example to get you started:

```python
import numpy as np
import pandas as pd
from PointQuality import QualityMetrics, GTA_Quality_Metrics

# Generate sample point cloud data
original_points = np.random.rand(100, 2)
reconstructed_points = original_points + np.random.normal(scale=0.05, size=(100, 2))

# Evaluate local quality metrics
qm = QualityMetrics(original_points, reconstructed_points)
metrics_dict = qm.evaluate_metrics(compute_distortion=True)
print(metrics_dict)
```


