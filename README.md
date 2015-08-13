### Getting Started with f15ece6504 

In this course, we will be using python considerably. Most of the assignments will need a good amount of python. Although many distributions of python are available, we recommend that you use the [Anconda Python](https://store.continuum.io/cshop/anaconda/) distribution. The advantage of using Anaconda is that: 
- It installs with most of the important [python-packages](http://docs.continuum.io/anaconda/pkg-docs)
- It does not need root access to install new packages 
- Supported by Linux, OS X and Windows
- It is completely free! 

We suggest that you use either Linux (prerferably Ubuntu) or OS X! 
Follow the instructions [here](http://docs.continuum.io/anaconda/install) to install Anaconda python. Remember to make Anaconda python the default python on your computer. Common issues are addressed here in the  [FAQ](http://docs.continuum.io/anaconda/faq). 

#### Python 
If you are comfortable with python, you can skip this section! 

If you are new to python and have sufficient programming experience in using languages like C/C++, MATLAB, etc., you should be able to grasp the basic workings of python necessary for this course easily. We will be using the [Numpy](http://www.numpy.org/) package extensively as it is the fundamental package for scientific computing providing support for array operations, linear algebra,etc. A good tutorial to get you started is [here](http://cs231n.github.io/python-numpy-tutorial/). For those comfortable with the operations of MATLAB, [this](http://sebastianraschka.com/Articles/2014_matlab_vs_numpy.html) might prove useful. 

For some assignments, we will be using the [ipython notebook](http://ipython.org/notebook.html). Ipython is a command shell for interactive computing developed primarily for python. The notebook is a useful environment where text can be embedded with code enabling us to set a flow while you do the assignments! If you have installed Anaconda and made it your default python, you should be able to start the ipython notebook envirnoment with:

```sh
$ ipython notebook
```
The ipython notebook files have .ipynb extension which you should be able to open now by navigating to the right directory. 

#### Starting homework 0:
This homework is a warm up for the rest of the course. As part of this homework you will be coding two classifiers: Support Vector Machine (SVM) and logistic regression. You will train these to classify images in the [CIFAR-10 dataset](http://www.cs.toronto.edu/~kriz/cifar.html). The CIFAR-10 is a toy dataset with 60000 images of size 32 X 32, belonging to 10 classes. You need to start with svm.ipynb first to implement the SVM and then go ahead with Softmax.ipynb to implement logistic regression. 

##### Getting the dataset
Make sure you are on the internet and navigate to hw0/f15ece6504/data in the hw0 folder. Run:
```sh
./get_datasets.sh
```
This script will download the python version of the database for you and put it in hw0/f15ece6504/data/cifar-10-batches.py folder. 

##### Getting Spearmint
As part of this homework, you will be using spearmint to tune the hyper-parameters like learning rate, regularization strength, etc. Spearmint is a software package to perform Bayesian optimization. It is designed to automatically run experiments in a manner that iteratively adjusts a number of parameters so as to minimize some objective in as few runs as possible. 

If you have anaconda installed, setting up spearmint should be pretty straightforward. You can find installation and usage instructions [here](https://github.com/HIPS/Spearmint). You need to use the command line interface to work with spearmint. You can look up the ../examples/branin/ to get an idea. 

> TLDR: Install spearmint! The rest is there on the ipython notebook. 

##### SVM and Logistic Regression
As you might already know, SVM and logistic regression clasisfiers are very similar except that they calculate the loss differently. Here is a brief summary of the classifiers and if you need a detailed tutorial to brush up your knowledge, [this](http://cs231n.github.io/linear-classify/) is a nice place!

Before we go into the details of a classifier, let us assume that our training dataset has $ x_i \in \R^D $ instances of dimensionality $D$. Corresponding to each of the training instances we have labels $y_i \in [1,K]$, where $K$ is the number of classes. In this homework, we are using the CIFAR-10 database where $N=50,000$, $K=10$, $D= 32 \times 32 \times 3$ (image of size (32,32) with three channels - Red, Green and Blue). 
Classification is the task of assigning a label to the input from a fixed set of categories or classes. A classifier consists of two important components:
1. Score function: This maps every input $x_i$ to a vector $p$ of dimensionality $K$. Each of these entries represent the class scores for that image. Both SVM and Logistic Regression have a linear score function given by:
$$
p_i = f(x_i;W,b)
$$
where, 
$$ 
f(x;W,b) = Wx + b
$$
Here, W is a matrix of weights of dimensionality $K \times D$ and b is a vector of bias terms of dimensionality $K$. The process of training is to find the appropriate values for W and b such that the score corresponding to the correct class is high. In order to do this, we need a function that evaluates the performance. Using this evaluation as feedback, the weigts can be updated in the right 'direction' to improve the performance of the classifier. 

We make a minor modification to the notation before proceeding. The bias term can be incorporated within the weight matrix W making it of dimensionality $K \times D+1$. The $i^{th}$ row of the weight matrix W is used as a column vector $w_i$ so that the $p_i^j = w_j^Tx_i$. The superscript j denotes the $j^th$ element of the score vector $p$. 

2. Loss function: This function quantifies the correspondence between the predicted scores and ground truth labels. 
The loss of the SVM is given by:
$$
L = \frac{1}{N}\sum_{i=1}^{N}\sum{j\neq y_i}max(0,w_j^Tx_i - w_{y_i}^Tx_i + \Delta)
$$
Here, \Delta is the margin. The loss function penalises when the correct class is not greater than all the other scores by atleast \Delta.

The loss of the Logistic Regression is given by:
$$
L = -\frac{1}{N}\sum{i=1}^{N}\log \bigg( \frac{e^{p_i^{y_i}}{\sum_j e^{p^j_i}} \bigg)
$$ 




A classifier has two important components:
1. Score Function: 
