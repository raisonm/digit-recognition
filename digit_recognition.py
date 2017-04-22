# first draft of the digit recognition neura net.
# Matthew raison...

import csv
import numpy
import math

def sigmoid(x):
	if x <= -10:
		return 0

	return 1.0 / (1.0 + math.exp(-x))

def d_sigmoid(z):
		return z * (1 - z);

def normalize_expected_digit_array(d, expect):
	if d == expect:
		return 1
	else:
		return 0

array_normalize_expected_digit_array = numpy.vectorize(normalize_expected_digit_array,  otypes=[numpy.float])
array_f = numpy.vectorize(sigmoid, otypes=[numpy.float]); # why a numpy float???
array_df = numpy.vectorize(d_sigmoid,  otypes=[numpy.float])

# training_set = csv.reader(open("train.csv"), delimiter=",");
training_set = numpy.genfromtxt("train.csv", delimiter=',');

is_training = 0;
training_set = training_set[1:,:] # get rid of labels...


######
# a few assumptions:
# the number of hidden nodes will be the same as the number of input nodes.
# we will start with only one output node, training for digital recognition of the number "1"
# I don't think it matters what weights we start with so we'll start with just random numbers between 0 and 1
# To help normalize values I'll devide each x input by 255, the max value, so we start of with a similar sort of value between 0 and 1.
######

if is_training == 0:
	# input_set = 2;
	x_inputes = numpy.array(training_set).astype(float);
	
	d = x_inputes[:,0][numpy.newaxis].T
	generator_d = d;

	d = numpy.append(d, generator_d, 1);
	d = numpy.append(d, generator_d, 1);
	d = numpy.append(d, generator_d, 1);
	d = numpy.append(d, generator_d, 1);
	d = numpy.append(d, generator_d, 1);
	d = numpy.append(d, generator_d, 1);
	d = numpy.append(d, generator_d, 1);
	d = numpy.append(d, generator_d, 1);
	d = numpy.append(d, generator_d, 1);
	d = array_normalize_expected_digit_array(d, numpy.array([[0,1,2,3,4,5,6,7,8,9]]))

	x_inputes = x_inputes[:,1:];
	x_inputes = x_inputes * (0.0039)
	x_inputes = x_inputes - 0.5 ## scale it back some

	w1 = numpy.genfromtxt("w1_values.csv", delimiter=",");
	# there will be only one output node. there will also be n inputs to this one node as well.
	w2 = numpy.genfromtxt("w2_values.csv", delimiter=",");

	successes = 0 
	failures = 0
	i = 0
	print 'iteration cap: ' + str(x_inputes.shape[0])
	while i < x_inputes.shape[0]:
		x = x_inputes[i,:][numpy.newaxis] ## this will be the value that changes each time. we'll iterate through all the different sets of x inputs
		actual = d[i,:][numpy.newaxis]

		y1 = numpy.dot(x, w1);
		z1 = array_f(y1);
		y2 = numpy.dot(z1, w2);
		z2 = array_f(y2);

		diff = actual - z2;
		# print 'lets see how many are correct. where ever there is a 1 there\'s an error...'
		dist = numpy.absolute(diff);
		errors = numpy.count_nonzero(dist)
		failures += errors
		print 'iteration: ' + str(i);
		i += 1
	

	successes = d.shape[0]*d.shape[1] - failures
	ratio = (successes*1.0) / (d.shape[0]*d.shape[1] * 1.0)
	print ''
	print 'number of errors: ' + str(failures)
	print 'number of successes:' + str(successes)
	print 'percentage correct:' + str(ratio)
	# numpy.set_printoptions(threshold='nan');

	# print dist
	# incorrect = numpy.dot(diff.T, numpy.ones((diff.shape[0], 1)))

	# ratio = incorrect / diff.shape[0];
	
	# print "how many were wrong? " + str(incorrect) + " | actual answer:" + str(ratio);
	quit();


step_size = 1.0

x_inputes = numpy.array(training_set).astype(float);
d = x_inputes[:,0][numpy.newaxis].T
generator_d = d;

d = numpy.append(d, generator_d, 1);
d = numpy.append(d, generator_d, 1);
d = numpy.append(d, generator_d, 1);
d = numpy.append(d, generator_d, 1);
d = numpy.append(d, generator_d, 1);
d = numpy.append(d, generator_d, 1);
d = numpy.append(d, generator_d, 1);
d = numpy.append(d, generator_d, 1);
d = numpy.append(d, generator_d, 1);

## so to normalize I can just load in an array like [[0,1,2,3,4,5,6,7,8,9]] for the second parameter.
## I do still need to create d as an array of n by 10 though. there should be a column for every final layer synapse solution out put....thingy...
array_of_possible_solutions = numpy.array([[0,1,2,3,4,5,6,7,8,9]])
d = array_normalize_expected_digit_array(d, array_of_possible_solutions.shape[1])


x_inputes = x_inputes[:,1:] ## ignore first column. thats the one with the solutions.
x_inputes = x_inputes * (0.0039)
# x_inputes = x_inputes - 0.5 ## scale it back some
x_inputes = x_inputes / x_inputes.shape[1]


## initiate the values here
m = x_inputes.shape[1]; # number of hidden nodes
s1 = m;
w1 = numpy.random.uniform(-1.0, 1.0, (m, s1)); ## I THINK THIS IS THE ORDER BUT I SHOULD DOUBLE CHECK
s2 = d.shape[1] ## we'll need 10 different synapses on our last layer to generate all the solutions.
w2 = numpy.random.uniform(-1.0, 1.0, (s1, s2)); # here we get a 42000X10 matrix
## previously  was 1, d.shape[1]
dp_dz = numpy.ones((d.shape[0], d.shape[1]));

cumulative_product = 1;
i = 0;
## cumulative_product > 0.0005 and
while i < 50:
	cumulative_product = numpy.prod(numpy.absolute(dp_dz));
	print "iteration number: " + str(i);
	## ok ok I think I got it....use all the inputs to start so that those end up changing the w1 and w2 values...yes?
	## it'll optimize for all of the input sets and stuff...maybe

	## 1Xm array
	# x = x_inputes[i,:][numpy.newaxis] ## this will be the value that changes each time. we'll iterate through all the different sets of x inputs
	# actual = d[i,:][numpy.newaxis]
	x=x_inputes
	actual=d

	## nXs1 array
	y1 = numpy.dot(x, w1);
	## nXs1 array
	z1 = array_f(y1); ## at this point we've obtained the activation on all hidden layer synapses
	## nXs2 array
	y2 = numpy.dot(z1, w2);
	## nXs2 array
	z2 = array_f(y2); ## at this point we've obtained the activation on all final layer synapses
	# print z2.shape
	# print '------'
	# print actual.shape
	dp_dz = actual - z2 # should be a 1Xs2 array

	## The only reason this isn't a dot product is because the activation on the final layer synapse will be a single value for each system.
	## a new model will need to be consider for systems where the final layer has more than one synapse
	delta_w2 = dp_dz * array_df(z2);
	# print '--------------'
	# print 'dp_dz'
	# print dp_dz.shape
	# print '--------------'
	# print 'z2'
	# print z2.shape
	# print '--------------'
	# print 'delta_w2'
	# print delta_w2.shape

	## 1Xs1
	delta_w1 = numpy.dot(delta_w2, w2.T)
	# print '--------------'
	# print 'delta_w1'
	# print delta_w1.shape
	# print '--------------'

	dp_dw1 = numpy.dot(x.T, delta_w1)
	dp_dw2 = numpy.dot(z1.T, delta_w2)

	# print 'dp_dw1'
	# print dp_dw1.shape
	# print '-------------'
	# print 'dp_dw2'
	# print dp_dw2.shape
	# print '-------------'
	# print 'z1'
	# print z1.shape
	# print '-------------'
	# print 'x_inputes'
	# print x.shape
	# print '-------------'
	w2 += (1) * dp_dw2 * step_size ## we're going to have to double check the math here because now this seems a little funny but i'm too lazy to check it now
	w1 += (1) * dp_dw1 * step_size

	# print 'dp_dz'
	# print dp_dz.shape
	# print '-------------'
	# print dp_dz
	## this should write write the values of w as this becomes optimized so when this is exited everything is in an actionable state
	numpy.savetxt("w1_values.csv", numpy.asarray(w1), delimiter=",");
	numpy.savetxt("w2_values.csv", numpy.asarray(w2), delimiter=",");
	i += 1

quit()



