""" 
Simple linear Regression 
Python using sklearn 

"""

# improt library;
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# loading the dataset
data = pd.read_csv("C:/Users/kavif/Documents/Git/Machine_learning_python/Simple Linear Regression/data/Salary_Data.csv")
x_data = data.iloc[:,0:1].values
y_data =  data.iloc[:,1:].values
print("Imported dataset")

# ploting the data
def plot_data():
    plt.scatter(x_data,y_data , color= 'blue')
    plt.title("dataset")
    plt.xlabel("years")
    plt.ylabel('Salary')
    plt.show()
plot_data()

print("Fitting the Machine")
# fiting the Machine learning algorthim
machine = LinearRegression()
machine.fit(x_data, y_data)

print("Plotting the result")
# ploting the result;
def result_plot():
    plt.scatter(x_data,y_data , color= 'blue')
    plt.plot(x_data,machine.predict(x_data), color='red')
    plt.title("dataset")
    plt.xlabel("years")
    plt.ylabel('Salary')
    plt.show()

result_plot()
