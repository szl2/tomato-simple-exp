#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
import pandas as pd

from sklearn.metrics import confusion_matrix
import seaborn as sns


# In[2]:

result = []

sed_list = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]

for sed in sed_list:
    raw_ab = [-0.468156425,
              -0.469536665,
              -0.412237762,
              -0.474250576,
              -0.473564955,
              -0.567211055,
              -0.600552656,
              -0.59111378,
              -0.572317724,
              -0.563609467,
              0.883399209,
              0.834977578,
              1.007662835,
              0.959257583,
              0.901252408,
              0.89565628,
              0.19259012,
              0.026351351,
              0.074183976,
              0.143338954,
              -0.025984049,
              -0.415363698,
              -0.410076977,
              -0.400073475,
              -0.387591495,
              -0.391610738,
              -0.355336617,
              -0.177881412,
              -0.325116113,
              -0.347753186,
              -0.333648096,
              -0.42907676,
              -0.316281971,
              0.084465663,
              0.083790706,
              0.190656132,
              -0.17497779,
              -0.082743771,
              0.49616041,
              0.593888219,
              0.54477938,
              0.510420924,
              0.578094108,
              -0.037252619,
              -0.012594458,
              0.079597438,
              -0.160812117,
              -0.061984733,
              -0.551927037,
              -0.580558056,
              -0.575621444,
              -0.563951779,
              -0.550799087,
              0.887878788,
              0.872207328,
              1.020447907,
              0.989396035,
              0.937796521,
              0.892976589,
              0.44709389,
              0.408970976,
              0.465534805,
              0.451253482,
              0.41056,
              0.033465166,
              0.111253197,
              0.109356015,
              0.109548725,
              -0.029448622,
              0.273426573,
              0.358208955,
              0.325320513,
              0.333448157,
              0.325573315,
              0.265217391,
              0.327580797,
              0.418114603,
              0.420523139,
              0.467180051,
              0.275702734,
              0.406641816,
              0.548234438,
              0.621752042,
              0.618719611,
              0.620761825,
              0.641609719,
              ]


    # In[3]:


    len(raw_ab)


    # In[4]:


    min(raw_ab)


    # In[5]:


    max(raw_ab)


    # In[6]:


    label_list = []
    lycopene_concentration = []
    chlorophyll_concentration = []

    for ab in raw_ab:
        label = 0
        lycopene_con = 0
        chlorophyll_con = 0
        if ab <= -0.44:
            label = 1
            ratio = (ab + 0.600552656)/(-0.44 + 0.600552656)
            lycopene_con = 0 + (2 - 0) * ratio
            chlorophyll_con = 50 + (45 - 50) * ratio
        elif -0.44 < ab <= -0.3:
            ratio = (ab + 0.44)/(-0.3+0.44)
            lycopene_con = 2 + (5 - 2) * ratio
            chlorophyll_con = 45 + (27 - 45) * ratio
            if ratio < 0.5:
                label = 1
            else:
                label = 2
        elif -0.3 < ab <= -0.03:
            ratio = (ab + 0.3)/(-0.03+0.3)
            lycopene_con = 5 + (10 - 5) * ratio
            chlorophyll_con = 27 + (25 - 27) * ratio
            if ratio < 0.5:
                label = 2
            else:
                label = 3
        elif -0.03 < ab <= 0.27:
            ratio = (ab + 0.03)/(0.27+0.03)
            lycopene_con = 10 + (23 - 10) * ratio
            chlorophyll_con = 25 + (12 - 25) * ratio
            if ratio < 0.5:
                label = 3
            else:
                label = 4
        elif 0.27 < ab <= 0.54:
            ratio = (ab - 0.27)/(0.54-0.27)
            lycopene_con = 23 + (32 - 23) * ratio
            chlorophyll_con = 12 + (10 - 12) * ratio
            if ratio < 0.5:
                label = 4
            else:
                label = 5
        elif 0.54 < ab <= 0.8:
            ratio = (ab - 0.54)/(0.8-0.54)
            lycopene_con = 32 + (52 - 32) * ratio
            chlorophyll_con = 10 + (5 - 10) * ratio
            if ratio < 0.5:
                label = 5
            else:
                label = 6
        elif 0.8 < ab:
            ratio = (ab - 0.8)/(1.007662835 - 0.8)
            lycopene_con = 52 + (70 - 52) * ratio
            chlorophyll_con = 5 + (2 - 5) * ratio
            label = 6

        for i in range(4):
            label_list.append(label)
            lycopene_concentration.append(lycopene_con)
            chlorophyll_concentration.append(chlorophyll_con)


    # In[7]:


    len(label_list)


    # # 1. Lycopene
    #
    # ## 1.1. load the data

    # In[8]:


    data_path_root = './cache/'


    # In[9]:


    cache_dir = data_path_root + 'Tomato3/PCA_input.npy'
    sample_data = np.load(cache_dir)
    sample_data


    # In[10]:


    num_wl = 8
    num_sample = 16
    num_area_list = np.array([20, 20, 24, 20, 20, 24, 24, 20, 20, 20, 24, 20, 20, 24, 24, 20])
    num_class = 1

    size = np.sum(num_area_list)
    train_size = size//10 * 8
    test_size = size - train_size


    # In[11]:


    train_size


    # In[12]:


    test_size


    # In[13]:


    np.sum(num_area_list)


    # In[14]:


    size/6


    # In[15]:


    X = np.zeros((8, size))
    Y = np.array(label_list)
    num = 0

    for sample_idx in range(num_sample):
        sample_idx += 1

        cache_dir = data_path_root + 'Tomato' + str(sample_idx) + '/PCA_input.npy'

        sample_data = np.load(cache_dir)


    #     print("-----------")
    #     print(sample_idx)
    #     print("-----------")

        for i in range(num_area_list[sample_idx-1]):
            X[:, num:(num + num_class)] = sample_data[i,:,:]
    #         print(sample_data[i,:,:] )


    #         Y[num:(num + num_class)] = list(set(range(num_class)))
            num = num + num_class


    # In[16]:


    X


    # In[17]:


    Y


    # In[18]:




    # In[19]:



    print("\n\n---------\n", sed)
    np.random.seed(sed)

    train_idx = []
    test_idx = []
    for i in range(1,7):
        (temp, ) = np.where(Y == i)
        temp_size = len(temp)
        train_size = 20
        test_size = 8

        # for train

        rand_select = np.random.choice(temp_size, train_size, replace=False)

        for idx in temp[rand_select]:
            train_idx.append(idx)


        # for test

        left_over = np.array(list(set(temp) - set(temp[rand_select])))

        leftover_size = len(left_over)

        rand_select = np.random.choice(leftover_size, test_size, replace=False)

        for idx in left_over[rand_select]:
            test_idx.append(idx)

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)


    # In[ ]:





    # In[20]:



    # train_idx = np.random.choice(size, train_size, replace=False)
    # test_idx = np.array(list(set(range(size)) - set(train_idx)))

    RGB_idx = np.array([0,3,7])

    X_train = X[:, train_idx].T
    X_train_rgb = X[RGB_idx][:, train_idx].T
    Y_train = Y[train_idx]

    X_test = X[:,test_idx].T
    X_test_rgb = X[RGB_idx][:,test_idx].T
    Y_test = Y[test_idx]


    # In[21]:



    # In[22]:


    X_test.shape


    # ## 1.2. load the model
    # ### 1.2.1 Logistic Regression


    print("\nCLASSIFICATION \n Logistic Regression")
    # In[23]:



    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    import warnings
    warnings.filterwarnings('ignore')


    # In[24]:


    # Full
    model_LR_1 = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter = 100)
    model_LR_1.fit(X_train, Y_train)
    y_pred = model_LR_1.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)
    print(f"full_Accuracy: {accuracy:.2f}")

    result.append(accuracy)


    # RGB
    model_LR_2 = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter = 100)
    model_LR_2.fit(X_train_rgb, Y_train)
    y_pred = model_LR_2.predict(X_test_rgb)
    accuracy = accuracy_score(Y_test, y_pred)
    print(f"rgb_Accuracy: {accuracy:.2f}")

    result.append(accuracy)




    print("\nKNN")

    # ### 1.2.2. KNN

    # In[26]:


    from sklearn.neighbors import KNeighborsClassifier


    # In[27]:


    # Create a k-Nearest Neighbors classifier for multi-class classification
    k = 5  # Number of neighbors

    # Full
    model_KNN_1 = KNeighborsClassifier(n_neighbors=k)
    model_KNN_1.fit(X_train, Y_train)
    y_pred = model_KNN_1.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)
    print(f"full_Accuracy: {accuracy:.2f}")

    result.append(accuracy)

    # RGB
    model_KNN_2 = KNeighborsClassifier(n_neighbors=k)
    model_KNN_2.fit(X_train_rgb, Y_train)
    y_pred = model_KNN_2.predict(X_test_rgb)
    accuracy = accuracy_score(Y_test, y_pred)
    print(f"rgb_Accuracy: {accuracy:.2f}")

    result.append(accuracy)


    print("\nSVM")

    # ## 1.2.3. SVM

    # In[29]:


    from sklearn.svm import SVC


    # In[30]:


    # Full
    model_SVM_1 = SVC(kernel='linear', C=1.0)  # You can experiment with different kernels and C values
    model_SVM_1.fit(X_train, Y_train)
    y_pred = model_SVM_1.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)
    print(f"full_Accuracy: {accuracy:.2f}")

    result.append(accuracy)

    cm = confusion_matrix(Y_test, y_pred)



    # RGB
    model_SVM_2 = SVC(kernel='linear', C=1.0)  # You can experiment with different kernels and C values
    model_SVM_2.fit(X_train_rgb, Y_train)
    y_pred = model_SVM_2.predict(X_test_rgb)
    accuracy = accuracy_score(Y_test, y_pred)
    print(f"rgb_Accuracy: {accuracy:.2f}")

    result.append(accuracy)


    print("\n\nREGRESSION \n Partial Least Square")


    # # Partial Least Square

    # In[31]:


    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score


    # ## Chlorophyll

    # In[32]:


    Y = np.array(chlorophyll_concentration)

    RGB_idx = np.array([0,3,7])

    X_train = X[:, train_idx].T
    X_train_rgb = X[RGB_idx][:, train_idx].T
    Y_train = Y[train_idx]

    X_test = X[:,test_idx].T
    X_test_rgb = X[RGB_idx][:,test_idx].T
    Y_test = Y[test_idx]


    # In[33]:


    Y_train.shape


    # In[34]:


    # Initialize the PLS regression model with a specified number of components
    n_components = 2
    pls = PLSRegression(n_components=n_components)

    # Fit the model on the training data
    pls.fit(X_train, Y_train)

    # Predict the target variable on the test data
    y_pred = pls.predict(X_test)

    print("\nChlorophyll-FULL")
    # Calculate and print the mean squared error on the test set
    mse = mean_squared_error(Y_test, y_pred, squared=False)
    print(f'Mean Squared Error: {mse}')

    result.append(mse)

    r2 = r2_score(Y_test, y_pred)
    print(f'R^2 Score: {r2}')

    result.append(r2)



    # In[35]:


    # Initialize the PLS regression model with a specified number of components
    n_components = 2
    pls = PLSRegression(n_components=n_components)

    # Fit the model on the training data
    pls.fit(X_train_rgb, Y_train)

    # Predict the target variable on the test data
    y_pred = pls.predict(X_test_rgb)

    y_pred = y_pred.flatten()

    print("\nChlorophyll-RGB")
    # Calculate and print the mean squared error on the test set
    mse = mean_squared_error(Y_test, y_pred, squared=False)
    print(f'Mean Squared Error: {mse}')

    result.append(mse)

    r2 = r2_score(Y_test, y_pred)
    print(f'R^2 Score: {r2}')

    result.append(r2)



    # ## Lycopene

    # In[36]:


    Y = np.array(lycopene_concentration)

    RGB_idx = np.array([0,3,7])

    X_train = X[:, train_idx].T
    X_train_rgb = X[RGB_idx][:, train_idx].T
    Y_train = Y[train_idx]

    X_test = X[:,test_idx].T
    X_test_rgb = X[RGB_idx][:,test_idx].T
    Y_test = Y[test_idx]


    # In[37]:


    Y_test


    # In[38]:


    Y_train.shape


    # In[39]:


    # Initialize the PLS regression model with a specified number of components
    n_components = 2
    pls = PLSRegression(n_components=n_components)

    # Fit the model on the training data
    pls.fit(X_train, Y_train)

    # Predict the target variable on the test data
    y_pred = pls.predict(X_test)

    print("\nLycopene-FULL")

    # Calculate and print the mean squared error on the test set
    mse = mean_squared_error(Y_test, y_pred, squared=False)
    print(f'Mean Squared Error: {mse}')

    result.append(mse)

    r2 = r2_score(Y_test, y_pred)
    print(f'R^2 Score: {r2}')

    result.append(r2)




    # In[40]:


    # Initialize the PLS regression model with a specified number of components
    n_components = 2
    pls = PLSRegression(n_components=n_components)

    # Fit the model on the training data
    pls.fit(X_train_rgb, Y_train)

    # Predict the target variable on the test data
    y_pred = pls.predict(X_test_rgb)

    print("\nLycopene-RGB")

    # Calculate and print the mean squared error on the test set
    mse = mean_squared_error(Y_test, y_pred, squared=False)
    print(f'Root Mean Squared Error: {mse}')

    result.append(mse)

    r2 = r2_score(Y_test, y_pred)
    print(f'R^2 Score: {r2}')

    result.append(r2)

    # In[ ]:



result = np.array(result)

result = result.reshape((len(sed_list),14))

print(result)

print(np.average(result, axis=0))


from numpy import asarray
from numpy import savetxt

data = asarray(result)

savetxt('data.csv', data, delimiter=',')