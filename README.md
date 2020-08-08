[![MIT License](https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=flat-square)](https://github.com/cch1999/protein-stability/blob/master/LICENSE)
[![LinkedIn](https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555)](https://www.linkedin.com/in/charlie-harris-388285156/)

# Distance Matrix - Convolutional Neural Networks

A simple implementation of enzyme function prediction by treating inter-residue distance matrices as images that are used to train a Convolutional Neural Network (CNN). Models are implemented in both in Keras and PyTorch.

![Matrix](https://github.com/cch1999/DMCNN/blob/master/figs/precomputed_single_channel.png)

An example distance matrix, presence of distinct proteins domains are observed as edges in the matrix "image".

## Results

Program aims to predict the first level of EC classification.

![Results](https://github.com/cch1999/DMCNN/blob/master/figs/accuracies.png)

### Prerequisites

Python packages used for this project

```
pytorch
tensorflow/keras
biopython
matplotlib
pandas
numpy
```

## Author

* **Charles Harris** - [cch1999](https://github.com/cch1999)


## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgements

Dataset was sourced from [EnzyNet](https://github.com/shervinea/enzynet), a similar approach that uses 3DCNNs.

