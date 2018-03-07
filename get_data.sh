#!/bin/bash
mkdir dataset
cd dataset
wget 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
echo "Downloaded; extracting dataset."
tar -xf cifar-10-python.tar.gz
echo "Done extracting dataset."
