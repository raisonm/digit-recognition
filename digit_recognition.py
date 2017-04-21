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

# training_set = csv.reader(open("train.csv"), delimiter=",");
training_set = numpy.genfromtxt("train.csv", delimiter=',');

is_training = 0;
# the first row is just going to be the labels for the columns...so don't really do anything with them at this point.
# first_row = next(training_set);
training_set = training_set[1:,:] # get rid of labels...

digit_range = [0,1,2,3,4,5,6,7,8,9];


######
# a few assumptions:
# the number of hidden nodes will be the same as the number of input nodes.
# we will start with only one output node, training for digital recognition of the number "1"
# I don't think it matters what weights we start with so we'll start with just random numbers between 0 and 1
# To help normalize values I'll devide each x input by 255, the max value, so we start of with a similar sort of value between 0 and 1.
######

x_inputes = numpy.array([[]]);

w1 = numpy.array([[]]);
w2 = numpy.array([[]]);

p = 1;

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
	array_f = numpy.vectorize(sigmoid, otypes=[numpy.float]); # why a numpy float???

	y1 = numpy.dot(x_inputes, w1);
	z1 = array_f(y1);
	y2 = numpy.dot(z1, w2);
	z2 = array_f(y2);

	diff = d - z2;
	# print 'lets see how many are correct. where ever there is a 1 there\'s an error...'
	dist = numpy.absolute(diff);
	# print ''
	errors = numpy.count_nonzero(dist)
	total = dist.shape[0] * dist.shape[1]
	success = total - errors
	ratio = (success * 1.0) / (total * 1.0)
	print ''
	print 'total cells: ' + str(total)
	print 'number of errors: ' + str(errors)
	print 'number of successes:' + str(success)
	print 'percentage correct:' + str(ratio)
	# numpy.set_printoptions(threshold='nan');

	# print dist
	# incorrect = numpy.dot(diff.T, numpy.ones((diff.shape[0], 1)))

	# ratio = incorrect / diff.shape[0];
	
	# print "how many were wrong? " + str(incorrect) + " | actual answer:" + str(ratio);
	quit();


step_size = 1

# first_row = training_set.shape[1] ## this should get the number of columns, or rather then umber of x inputs each run

array_f = numpy.vectorize(sigmoid, otypes=[numpy.float]); # why a numpy float???
array_df = numpy.vectorize(d_sigmoid,  otypes=[numpy.float])

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
x_inputes = x_inputes - 0.5 ## scale it back some


## initiate the values here
m = x_inputes.shape[1]; # number of hidden nodes
s1 = m;
w1 = numpy.random.uniform(-1.0, 1.0, (m, s1)); ## I THINK THIS IS THE ORDER BUT I SHOULD DOUBLE CHECK
s2 = d.shape[1] ## we'll need 10 different synapses on our last layer to generate all the solutions.
w2 = numpy.random.uniform(-1.0, 1.0, (s1, s2)); ## here we get a 42000X10 matrix
dp_dz = numpy.ones((1, d.shape[1]));


i = 0;
while numpy.amax(dp_dz) > 0.05 and i < 10:
	print "iteration number: " + str(i);
	i += 1

	y1 = numpy.dot(x_inputes, w1);
	z1 = array_f(y1);
	y2 = numpy.dot(z1, w2);
	z2 = array_f(y2);
	print z2.shape
	print '------'
	print d.shape
	dp_dz = d - z2 # at this point this will be 42000X10
	# print "error is: " + str(dp_dz);
	# print dp_dz.shape
	# print numpy.amax(dp_dz)

	## The only reason this isn't a dot product is because the activation on the final layer synapse will be a single value for each system.
	## a new model will need to be consider for systems where the final layer has more than one synapse
	delta_w2 = dp_dz * array_df(z2);
	print '--------------'
	print 'dp_dz'
	print dp_dz.shape
	print '--------------'
	print 'z2'
	print z2.shape
	print '--------------'
	print 'delta_w2'
	print delta_w2.shape

	# delta_1 = delta_1[0][0];
	# print delta_1.shape;

	delta_w1 = numpy.dot(delta_w2, w2.T)
	print '--------------'
	print 'delta_w1'
	print delta_w1.shape
	print '--------------'

	dp_dw1 = numpy.dot(x_inputes.T, delta_w1)
	dp_dw2 = numpy.dot(z1.T, delta_w2)

	print 'dp_dw1'
	print dp_dw1.shape
	print '-------------'
	print 'dp_dw2'
	print dp_dw2.shape
	print '-------------'
	print 'z1'
	print z1.shape
	print '-------------'
	print 'x_inputes'
	print x_inputes.shape
	print '-------------'
	# print dp_dw1.shape
	# print 
	w2 += dp_dw2 * step_size ## we're going to have to double check the math here because now this seems a little funny but i'm too lazy to check it now
	w1 += dp_dw1 * step_size

	print dp_dz
	## this should write write the values of w as this becomes optimized so when this is exited everything is in an actionable state
	numpy.savetxt("w1_values.csv", numpy.asarray(w1), delimiter=",");
	numpy.savetxt("w2_values.csv", numpy.asarray(w2), delimiter=",");

quit()

# print first_row.size
# quit();

# while abs(p) > 0.0000000000000000000001: # we'll say this is the threshold for accurate predictions
	# print first_row[1];
# for i in training_set:

# 	# i = next(training_set)
# 	# d is the actual digit represented in each row....um we may want to figure out how to properly quantify this though...
# 	# I'd suggest making it a boolean and have different branches of the neural net for each posible digit [0-9]
# 	# that way d is either just 0 or 1; Either the sinode at the end of the net fired or it didn't.
# 	digit_is_actually_one = i[0] == "1";

# 	d1 = 0; # this value should stand for 1:yes this is a match. we should expect our neural net to calculate a match.
	
# 	# lets only work on those rows that are going to train for the digit "1"
# 	if digit_is_actually_one:
# 		d1 = 1;


# 	# print type(i);
# 	x_inputes = numpy.array([i]).astype(float);
# 	x_inputes[0][0] = -1.0; ## replace the old d value with a -1. This well remove the irrelevent variable and add the threshold.
# 	x_inputes = numpy.transpose(x_inputes);
# 	#################################################
# 	# run the neural net recording values for everything. We'll want to put this into its own separate function.

# 	# def run_net(x, d1, ): 

# 	# 1/255 = 0.0039 approximately, and we want to scale all the values to some degree to start with....I think...
# 	# though this could be built into the w1 values, scaling now allows us to better ensure the x values are independent of eachother
# 	x_inputes = x_inputes * (0.0039);

# 	# n = x_inputes.size; # number of hidden nodes
# 	# m = n;

# 	# w1 = numpy.random.uniform(-1.0, 1.0, (m, n)); ## I THINK THIS IS THE ORDER BUT I SHOULD DOUBLE CHECK

# 	# print "done";
# 	# print x_inputes[0][0];

# 	# print x_inputes[0][0];

# 	# lets use y's to denote the summed values of the weights with their inpute values before they enter the sigmoid
# 	y1 = numpy.dot(x_inputes.T, w1);
# 	# print y1[0][0]

# 	array_f = numpy.vectorize(sigmoid, otypes=[numpy.float]); # why a numpy float???
# 	# And lets use z's to denote the value after the y's have been passed throught the sigmoid.
# 	z1 = array_f(y1).T;
# 	# print z1

# 	# there will be only one output node. there will also be n inputs to this one node as well.
# 	# w2 = numpy.random.uniform(-1.0, 1.0, (n, 1));
# 	# print w2;
# 	# print z1.size;

# 	y2 = numpy.dot(z1.T, w2);

# 	z2 = array_f(y2).T;
# 	# return d1 - z2;


# 	################################################
# 	# now to calculate the gradient of what just happened up there...

# 	array_df = numpy.vectorize(d_sigmoid,  otypes=[numpy.float])
# # here's the issue. I use f, not f prime here.
# 	dp_dw2 = numpy.dot(numpy.ones((1, n)), array_df(z1)) * (d1 - z2[0][0])

# 	df = array_df(z1); 

# 	dp_dw1 = numpy.dot(w2.T, df);
# 	# print numpy.ones((1, n))
# 	dp_dw1 = numpy.dot(dp_dw1, numpy.ones((1, n)));
# 	dp_dw1 = numpy.dot(dp_dw1, x_inputes);
# 	dp_dw1 = dp_dw1 * (d1 - z2[0][0])

# 	# print dp_dw2 + dp_dw1;
# 	p = d1 - z2;
# 	gradient = dp_dw2 + dp_dw1;

# 	w2 = w2 - (gradient * step_size);
# 	w1 = w1 - (gradient * step_size);

# 	print "p: " + str(p) + " | gradient: " + str(gradient)
# 	# break;
# 	# dp_dw1 = w2.T

# 	# z2 is our answer for this run.


# numpy.savetxt("w1_values.csv", numpy.asarray(w1), delimiter=",");
# numpy.savetxt("w2_values.csv", numpy.asarray(w2), delimiter=",");

# quit()

