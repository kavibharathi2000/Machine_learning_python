"""
Logistic Regression Scratch Python

"""
# importing library
import pandas as pd
import numpy as np 

# class Logistic Regression 
class Logistic_Regression:

    def __init__(self, learning_rate , epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = None

    # Sigmoid function
    def Sigmoid(self,input):
        output =1/(1+ np.exp(-input))
        return output

    # fitting the dataset
    def fit(self , x, y):
        n_samples , n_features = x.shape
        
        # defining the bias and weight
        self.w = np.zeros((n_features,1))
        self.b = np.zeros(1)

        for _ in range(self.epochs):
            # predicting (y= mx+c)
            y_pred = np.dot(x,self.w) + self.b
            pred = self.Sigmoid(y_pred)
            
            # cost calculation
            cost = -(1/n_samples)* np.sum( (y*np.log(pred)) + (1-y)*np.log(1-pred))
            print(">> Cost :", cost)

            # Gradient Descent algorthim
            # derivatives
            dw = (1/n_samples) * np.sum( np.dot(x.T,(pred-y)) )
            db = (1/n_samples) * np.sum(pred-y)

            # updating the parameters
            self.w = self.w - self.learning_rate*dw
            self.b =  self.b - self.learning_rate *db

    # predict function
    def predict(self,x):
        output = np.dot(x,self.w)+ self.b
        return output

    # function evaluation
    def eval(self,pred ,act):
        for i in range(len(act)):
            if(pred[i]>0):
                print("1", "==", act[i])
            else:
                print("0", "==", act[i])


# main function
if __name__ == "__main__":
    # loading the dataset
    data = pd.read_csv("data\Social_Network_Ads.csv")
    x_data = data.iloc[:,0:2].values
    y_data = data.iloc[:,2:3].values
    
    # data preprocessing
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x_data = sc.fit_transform(x_data)

    # train test split
    x_train = x_data[0:300,:]
    x_test = x_data[300:,:]
    y_train = y_data[0:300]
    y_test = y_data[300:]
    
    # machine parameters
    learning_rate = 0.0001
    epochs = 1000

    regressor = Logistic_Regression(learning_rate, epochs)
    regressor.fit(x_train , y_train)
    pred = regressor.predict(x_test)
    regressor.eval(pred,y_test)
    