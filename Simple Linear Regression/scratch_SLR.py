"""
Simple Linear Regression  - Scratch 
Python

"""
# importing required libraries
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


# Root mean square error  function
def rmse(actual, pred):
    sq_out=0
    for i in range(len(actual)):
        temp = pred[i] - actual[i]
        sq_out+= math.pow(temp,2)
    mean_output = sq_out/float(len(actual))
    output = math.sqrt(mean_output)
    return output


# variacne function
def variance(x):  
    for i in x:
        output =np.sum(math.pow((i- np.mean(x)),2))
    return output


#  covariance function
def covariance(x,y):
    for i in range(len(x)):
        output = np.sum( (x[i]-np.mean(x))*(y[i]- np.mean(y)) )
    return output


# calculating coefficient 
def coefficient(x,y):
    b1 = covariance(x,y)/variance(x)
    b0 = np.mean(y) - b1* np.mean(x)
    return b0,b1


# predict function
def predict(x,y):
    b0,b1 = coefficient(x,y)
    output = list()
    for i in x:
        y = (b1*i)+b0
        output.append(y)
    return output


# plotting the graph
def plot(x,y,line_plot):
    # plot with the output prediction line
    if line_plot:
        plt.scatter(x,y, color='red')
        plt.plot(x, predict(x,y), color="blue")
        plt.xlabel("Experience")
        plt.ylabel("Salary")
        plt.show()

    # simply plot the dataset
    else:
        plt.scatter(x,y, color='red')
        plt.xlabel("Experience")
        plt.ylabel("Salary")
        plt.show()


# simple linear regression function
def Simple_linear_Regression(x,y):
    plot(x,y,False)
    y_pred = predict(x,y)
    error = rmse(y, y_pred)
    print("The error of the machine is :", error)
    plot(x,y,True)
    
# main function
if __name__ == "__main__":

    # loading the dataset
    data_file = pd.read_csv("/Simple Linear Regression/data/Salary_Data.csv")
    x_data = data_file.iloc[:,0:1].values
    y_data = data_file.iloc[:,1:].values

    # simple linear regression 
    Simple_linear_Regression(x_data,y_data)
