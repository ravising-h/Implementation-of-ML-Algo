#--------------------------
# Importing Libraries
#--------------------------
import numpy as np 

#----------------------
# Class Linear Regressssion
#-----------------------
class LinearRegression:

#----------------------
# Constructor __init__ self => ravi
#-----------------------
    def __init__(ravi,train,label, iters = 100,lr = 0.001, feature_Scaling = False):

    	'''
		Parameter
		train - np array ;features (X)
		label - np  array ; labels (Y)
		iters - int; number of iteration in BGD 
				default 100
		lr -   float; learning rate alpha
				default = 0.001
		features_scaling Boolen: scale the feature
						default True
		return:

		m - np array; coefficients m[0] = C 
		cost - np.array; cost variation

		Example:
		`from LinearRegression import LinearRegression`
		`regressor = LinearRegression(x,y,10000,0.005,False)`
		`theta, cost = regressor.fit()`
		`prediction = regressor.predict(test)`
    	'''

        ravi.x = train
        ravi.y = label
        ravi.iters = iters
        ravi.lr = lr
        ravi.feature_Scaling = feature_Scaling
        ravi.m = np.zeros((ravi.x.shape[1]+1))
        ravi.mean = np.ones(ravi.x.shape[1]+ 1)
        ravi.std = np.ones(ravi.x.shape[1] +1)
#----------------------
# feature scaling (x - mean(x)) / std
#-----------------------

    def Feature_Scaling_Data(ravi,x):
            for i in range(0, ravi.x.shape[1]):
                ravi.mean[i] = np.mean(ravi.x.transpose()[i])
                ravi.std[i] = np.std(ravi.x.transpose()[i])
                for j in range(0, ravi.x.shape[0]):
                    ravi.x[j][i] = (ravi.x[j][i] - ravi.mean[i])/ravi.std[i]
            return x
#----------------------
# setting arrays
#-----------------------

    def setting_values(ravi):
        if ravi.feature_Scaling  == True:
            ravi.x = ravi.Feature_Scaling_Data

        n_row = ravi.x.shape[0]                        # no of Data point in Training Set.
        n_col = ravi.x.shape[1] + 1                    # no of Parameter + bias M1 ,M2 ,M3.. + C.
        X0 = np.ones(shape = (n_row,1))                # Adding Bias to training Data.
        ravi.x = np.concatenate((X0,ravi.x), axis = 1) # C can be represented as X0 * M0 where X0 = 1 so C = M0 a constant.

        return ravi.x,n_row,n_col
#----------------------
# step grad
#-----------------------

    def hypothesis(ravi,m, x, n_col):
        wm = np.ones((ravi.x.shape[0],1))
        ravi.m = ravi.m.reshape(1,n_col)
        for i in range(0,ravi.x.shape[0]):
            wm[i] = float(np.matmul(ravi.m, ravi.x[i]))
        wm = wm.reshape(ravi.x.shape[0])
        return wm
#----------------------
# Main Batch Gradient Descent Function
#-----------------------

    def Gradient_Descent(ravi,x,y,m, wm):
        cost = np.zeros(shape = (ravi.iters))
        for iterr in range(ravi.iters):
            ravi.m[0] -= (ravi.lr/x.shape[0]) * sum(wm - ravi.y)
            for j in range(1,ravi.x.shape[1]):
                ravi.m[0,j] -= (ravi.lr/ravi.x.shape[0]) * sum((wm-y) * ravi.x.transpose()[j])                       
            wm = ravi.hypothesis(ravi.m, ravi.x, ravi.x.shape[1])
            cost[iterr] = (1/ravi.x.shape[0]) * 0.5 * sum(np.square(wm - y))
        ravi.m = ravi.m.reshape(1,ravi.x.shape[1])
        return ravi.m, cost
#----------------------
# Fit Fuction
#-----------------------


    def fit(ravi):
        ravi.x,n_row,n_col = ravi.setting_values()
        wm = ravi.hypothesis(m, ravi.x, n_col)
        ravi.m, cost = ravi.Gradient_Descent(ravi.x,ravi.y, m ,wm)
        return ravi.m,cost
#----------------------
#
#-----------------------

    def predict(ravi, test):
        if ravi.feature_Scaling == True:
            test = feature_Scaling(test)
        x0 = np.ones((test.shape[0],1))
        test = np.concatenate((x0,test), axis =1 )
        print(test.shape,m)
        return np.array([np.matmul(ravi.m, test[i].T) for i in range(test.shape[0])])
