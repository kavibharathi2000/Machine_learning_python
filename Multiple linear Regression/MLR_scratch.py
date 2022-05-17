
# importing library
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


# Multiply Linear Regression class

class Multiple_Linear_Regression:
    def __init__(self, learning_rate, epochs):
        self.theta = None
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.cost_list = None
    
    # fitting the data 
    def fit(self, x,y):
        
        self.theta = np.zeros((x.shape[1],1))  
        self.cost_list = list()
        for _ in range(self.epochs):
            y_pred = np.dot(x,self.theta)
            
            # Cost value 
            cost = (1/2*y.shape[1]) * np.sum( np.square(y_pred - y))

            if (_%10 ==0):
                #print(">> Cost :", cost)
                self.cost_list.append(cost)

            # Gradient Descent 
            d_theta = (1/y.shape[1]) * np.dot(x.T, (y_pred - y))

            # updating  theta 
            self.theta = self.theta - self.learning_rate * d_theta
        
    # prediction function:
    def predict(self, x):
        output = np.dot(x,self.theta)
        return output
    
    # evalution function
    def eval(self,pred,actual, graph):
        
        # plotting the cost graph with respect to epochs
        if(graph):
            count = np.arange(0,len(self.cost_list))
            plt.plot(count, self.cost_list, color="red")
            plt.title("Machine Evaluation")
            plt.xlabel("count")
            plt.ylabel("cost")
            plt.show()
        
        # displaying the result of comparision of predicted and actual
        for i in range(len(pred)):
            print(pred[i]," == ", actual[i])





# main function
if __name__ =="__main__":
    
    # importing the datafile
    data = pd.read_csv("C:/Users/kavif/Documents/Git/Machine_learning_python/Multiple linear Regression/data/50_Startups.csv")
    x_value= data.iloc[:,0:3].values
    y_value = data.iloc[:,4:].values

    # data preprocessing
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x_data = sc.fit_transform(x_value)
    y_data = sc.fit_transform(y_value)

    # train and test data
    x_train = x_data[0:40,:]
    x_test = x_data[40:,:]
    y_train = y_data[0:40,:]
    y_test = y_data[40:,:]

    # parameters
    learning_rate = 0.0001
    epoch = 10000
    
    # Multiple linear regression a
    machine = Multiple_Linear_Regression(learning_rate , epoch)
    machine.fit(x_train, y_train)
    predicted = machine.predict(x_test)
    machine.eval(predicted,y_test,graph=True)