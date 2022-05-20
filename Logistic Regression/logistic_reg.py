
# importing library
import pandas as pd
import numpy as np


# main function
if __name__ == "__main__":
    data= pd.read_csv("data\Social_Network_Ads.csv")

    x_value = data.iloc[:,0:2].values
    y_value =data.iloc[:,2:3].values

    from sklearn.model_selection import train_test_split
    x_trains , x_tests , y_train , y_test = train_test_split(x_value , y_value , test_size=0.2, random_state=0)



    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x_train = sc.fit_transform(x_trains)
    x_test = sc.fit_transform(x_tests)


    from sklearn.linear_model import LogisticRegression
    engine = LogisticRegression()
    engine.fit(x_train , y_train)
    predcition =engine.predict(x_test)

    for i in range(len(predcition)):
        print(predcition[i], " == ", y_test[i])
        