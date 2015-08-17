# function that is called by spearmint! This calls the classifier and returns the value that needs to be minimized
# you may choose to return either the loss or the error on the validation set.

# we always do this! :/
import numpy as np

# import whichever classifier you need: LinearSVM/Softmax
# Uncomment below line for SVM
#from f15ece6504.classifiers import LinearSVM
# Uncomment below line for Softmax
#from f15ece6504.classifiers import Softmax

# see load_cifar10_tvt.py in the hw0/ folder. This code helps you get your data in a single shot! 
# This is basically a function that has all the steps you did in the ipython notebook to ready your data 
# for the classifier
from load_cifar10_tvt import load_cifar10_train_val

def get_valError(learning_rate,reg):
	
	# load data
	X_train,y_train,X_val,y_val = load_cifar10_train_val()
	# init the classifier you need
	
	# Uncomment below line for SVM
	# classifier = LinearSVM() 
	
	# Uncomment below line for SVM
	# classifier = Softmax() 

	# train classifier
	loss_hist = classifier.train(X_train, y_train, learning_rate=1e-7, reg=5e4,num_iters=1500, verbose=True)
	# get validation error
	y_val_pred = classifier.predict(X_val)
        val_accuracy = np.mean(y_val == y_val_pred)
	# return error rate
	return (1 - val_accuracy)

# Write a function like this called 'main'! 
def main(job_id, params):
    print 'Anything printed here will end up in the output directory for job #%d' % job_id
    print params
    return get_valError(params['learning_rate'], params['reg'])
	
	


