
# CIFAR-10 image classification with Lasagne

See the `cifar_lasagne.ipynb` IPython notebook. The "v3" network gets
84% classification accuracy on the validation set.

## Data

Get CIFAR-10 data in Python format and extract into `cifar-10-batches-py`:
```
wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar xzf cifar-10-python.tar.gz
```

## Acknowledgement

Based on Andrej Karpathy's excellent [CS231n course](http://cs231n.github.io/) materials.
