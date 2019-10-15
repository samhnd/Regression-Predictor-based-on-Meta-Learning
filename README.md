# A Meta-Learning Approach to Implement a Machine Learning Algorithm Selection Engine

Meta-learning comprises the concept of learning about learning. That is, meta-learning can provide the meta-knowledge obtained from the algorithms’ learning process to recommend algorithms to use a specific given dataset. 

Many algorithms have their own characteristics and it takes time to try every models for a given dataset.

Thus, this project aims to use meta-learning techniques to implement a recommendation engine to handle the challenge of machine learning algorithm selection. From a dataset, the engine will be able to recognize which algorithm suits the best the dataset and from which family it came from.

However, the implemented engine won't have knowledge about  every domains and won't be adapted to automatic decision. It will also need a dataset with a value to target, it cannot resolve deep learning problems.

#### Command to use the library:

```
from automl import prediction
```

#### Examples:

Tests were presented in the following notebooks:
- Boston House dataset
- Cancer dataset

