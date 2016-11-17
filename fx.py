import random
from numpy import linalg as LA
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import data
np.seterr(all='ignore')


class Nueral_Network(object):
	def __init__(self, Lambda=0):
		#define parameters
		self.inputLayerSize = 2
		self.outputLayerSize = 1
		self.hiddenLayerSize = 4

		#Weights
		self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
		self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

		#regularisation parameter
		self.Lambda = Lambda

	def forward(self, inputMatrix):
		#propogate inputs through network
		self.z2 = np.dot(inputMatrix, self.W1)
		self.a2 = self.sigmoid(self.z2)
		self.z3 = np.dot(self.a2, self.W2)
		prediction = self.sigmoid(self.z3)
		return prediction
	
	def sigmoid(self, z):
		#applies sigmoid activation function
		return 1/(1+np.exp(-z))

	def sigmoidPrime(self, z):
		#gradient of sigmoid from derivative of sigmoid
		return np.exp(-z)/((1+np.exp(-z))**2)

	def costFunction(self, X, y):
		#Computes cost for given input and result using pre-stored weigths
		self.p = self.forward(X)
		J = 0.5*sum((y-self.p)**2)/X.shape[0] + (self.Lambda/2)*(np.sum(self.W1**2)+np.sum(self.W2**2))

		return J

	def costFunctionPrime(self, X, y):
		#compute derivative
		self.p = self.forward(X)

		delta3 = np.multiply(-(y-self.p), self.sigmoidPrime(self.z3))
		dJdW2 = np.dot(np.transpose(self.a2), delta3)/X.shape[0] + self.Lambda*self.W2

		delta2 = np.dot(delta3, np.transpose(self.W2))*self.sigmoidPrime(self.z2)
		dJdW1 = np.dot(np.transpose(X), delta2)/X.shape[0] + self.Lambda*self.W1

		return dJdW1, dJdW2

	def getParams(self):
		#roll weights into one vector
		params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
		return params

	def setParams(self, params):
		#set W1 and W2 using single parameter vector
		W1_start = 0
		W1_end = self.hiddenLayerSize*self.inputLayerSize
		self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize, self.hiddenLayerSize))
		W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
		self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))

	def computeGradients(self, X, y):
		#computes gradients direct from cost function
		dJdW1, dJdW2 = self.costFunctionPrime(X, y)
		return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))


class trainer(object):
	def __init__(self, N):
		#local reference to NN
		self.N = N

	def costFunctionWrapper(self, params, X, y):
		self.N.setParams(params)
		cost = self.N.costFunction(X, y)
		grad = self.N.computeGradients(X, y)
		return cost, grad

	def callbackF(self, params):
		self.N.setParams(params)
		self.J.append(self.N.costFunction(self.X, self.y))
		self.testJ.append(self.N.costFunction(self.testX, self.testy))

	def train(self, trainX, trainy, testX, testy):
		#Internal variables for callback function
		self.X = trainX
		self.y = trainy

		self.testX = testX
		self.testy = testy

		#list to store costs
		self.J = []
		self.testJ = []

		params0 = self.N.getParams()

		options = {'maxiter' : 200, "disp" : True}
		_res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
                                 args=(trainX, trainy), options=options, callback=self.callbackF)

		self.N.setParams(_res.x)
		self.optimizationResults = _res

def computeNumericalGradient(N, X, y):
	#computes gradients numerically 
	paramsInitial = N.getParams()
	numgrad = np.zeros(paramsInitial.shape)
	perturb = np.zeros(paramsInitial.shape)
	zeroApprox = 1e-4

	for param in range(len(paramsInitial)):
		#set the perturbation vector
		perturb[param] = zeroApprox
		N.setParams(paramsInitial + perturb)
		loss2 = N.costFunction(X, y)

		N.setParams(paramsInitial - perturb)
		loss1 = N.costFunction(X, y)

		print((loss1 - loss2)/2*zeroApprox)
		#compute numerical gradient
		numgrad[param] = (loss2 - loss1)/(2*zeroApprox)

		#return to 0
		perturb[param] = 0

	#return to origanl parameters
	N.setParams(paramsInitial)

	return numgrad

NN = Nueral_Network(Lambda=0.00003)

trainX = data.trainX
trainy = data.trainy
testX = data.testX
testy = data.testy

T = trainer(NN)
T.train(trainX, trainy, testX, testy)

#Plot cost during training:
plt.plot(T.J)
plt.plot(T.testJ)
plt.grid(1)
plt.xlabel('Iterations')
plt.ylabel('Cost')

#Test network for various combinations of sleep/study:
GoldGBP = np.linspace(0, 1100, 100)
GoldUSD = np.linspace(0, 1800, 100)

#Normalize data (same way training data way normalized)
GoldGBPNorm = GoldGBP/1100.
GoldUSDNorm = GoldUSD/1800.

#Create 2-d versions of input for plotting
a, b  = np.meshgrid(GoldGBPNorm, GoldUSDNorm)

#Join into a single input matrix:
allInputs = np.zeros((a.size, 2))
allInputs[:, 0] = a.ravel()
allInputs[:, 1] = b.ravel()

allOutputs = NN.forward(allInputs)

#Contour Plot:
yy = np.dot(GoldUSD.reshape(100,1), np.ones((1,100)))
xx = np.dot(GoldGBP.reshape(100,1), np.ones((1,100))).T
CS = plt.contour(xx,yy, allOutputs.reshape(100, 100))
plt.clabel(CS, inline=1, fontsize=10)
plt.xlabel('Gold GBP')
plt.ylabel('Gold USD')


fig = plt.figure()
ax = fig.gca(projection='3d')
#scatter training examples
ax.scatter(1100*trainX[:,0], 1800*trainX[:,1], trainy, c='k', alpha = 1, s=30)
surf = ax.plot_surface(xx, yy, allOutputs.reshape(100, 100), \
                       cmap=plt.cm.jet, alpha = 0.5)

ax.set_xlabel('Gold GBP')
ax.set_ylabel('Gold USD')
ax.set_zlabel('GBPUSD')

print(ax.format_coord(1018.14, 1311.74))
plt.show()
