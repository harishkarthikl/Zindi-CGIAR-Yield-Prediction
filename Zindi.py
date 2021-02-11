#This files is a library of funtions used in Zindi CGIAR project.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import seed
seed(7)

def zindi_help():
    print("List of functions in this Zindi Utility Library\n")
    print("--INPUTS--")
    print("read_train_img(), read_test_img()\n")
    print("--PREPROCESSING--")
    print("std_scale_input(X), process_indices(X), clean_input(X, months, bands), shuffle(X, y),"
          "train_val_test_split(X, y, val_test size), robust_scale_input(X)\n")
    print("--OUTPUTS--")
    print("learning_plots(history callback), ready_sub(test_X, model, filename, batch_size=10)")

#Plot the learning curve (loss & metric)
def learning_plots(History):
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.plot(History.history['loss'])
    plt.plot(History.history['val_loss'])
    plt.grid(True)
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')

    plt.subplot(122)
    plt.plot(History.history['root_mean_squared_error'])
    plt.plot(History.history['val_root_mean_squared_error'])
    plt.grid(True)
    plt.title('Model RMSE')
    plt.ylabel('RMSE')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    plt.show()

#function to calculate various indices
def process_indices(X):
    # Calculate Normalized Difference Vegetation Index, NDVI = (B8 - B4) / (B8 + B4)
    B8 = X[:, :, :, :, 7]
    B4 = X[:, :, :, :, 3]
    NDVI = (B8 - B4) / (B8 + B4)

    # MODSAT Enhanced Vegetation Index, EVI = 2.5 * (B8 - B4) / ((B8 + 6.0 * B4 - 7.5 * B2) + 1.1) results in "div by zero" issue
    # Optional Sentinel 2A Enhanced Vegetation Index, EVI = 2.5 * (B8 - B4) / (B8 + (2.4 * B4) + 10000)
    # Lets stick with Sentinel 2A EVI2 = 2.4 * (B8 - B4) / (B8 + B4 + 1.0)
    B2 = X[:, :, :, :, 1]
    EVI2 = 2.4 * (B8 - B4) / (B8 + B4 + 1.0)

    # Calculate Moisture Index, MSI = B11 / B8
    B11 = X[:, :, :, :, 11]
    MSI = B11 / B8

    # Calculate Normalized Difference Moisture Index, NDWI = (B8 - B11) / (B8 + B11)
    NDWI = (B8 - B11) / (B8 + B11)

    # Creat new array to combine all indices and return the array
    X_ind = np.zeros((X.shape[0], 12, 40, 40, 4))
    X_ind[:, :, :, :, 0] = NDVI
    # Lets assign EVI to B5 channel.
    X_ind[:, :, :, :, 1] = EVI2
    # Lets assign MSI to B6 channel.
    X_ind[:, :, :, :, 2] = MSI
    # Lets assign NDWI to B7 channel.
    X_ind[:, :, :, :, 3] = NDWI

    show_stats(X_ind)
    print("NDVI, EVI2, MSI & NDWI added to channels 0, 1, 2 & 3 respectively!")
    print("Processed input shape is: ", X_ind.shape)
    return X_ind

#Keep necessary bands and remove unnnecessary ones
def clean_input(X, months=[0,9,10,11], bands=[]):
    # Delete months listed
    X = np.delete(X, months, 1)

    #create a dictory of bands
    bands_dict = {'B1': 0, 'B2': 1, 'B3': 2,'B4': 3, 'B5': 4,'B6': 5,'B7': 6,'B8': 7,'B8A': 8,
                  'B9': 9,'B10': 10,'B11': 11,'B12': 12,'QA60': 13, 'aet': 14, 'def': 15, 'pdsi': 16,
                  'pet': 17, 'pr': 18, 'ro': 19, 'soil': 20, 'srad': 21, 'tmmn': 22,
                'tmmx': 22, 'vap': 24, 'vpd': 25, 'vs': 26}
    # Keep only user supplied bands
    idx = []
    for band in bands:
        idx.append(bands_dict[band])
    X1 = X[:,:,:,:,idx].copy()
    print("Modified input shape is: ", X1.shape)
    show_stats(X1)
    return X1

#Function to scale input features (similar to sklearn standard scaler)
def std_scale_input(X):
    X_mean = np.mean(X, keepdims=True)
    X_std = np.std(X, keepdims=True)
    X_scaled = (X - X_mean) / X_std
    show_stats(X_scaled)
    print("Features of input array have been scaled!")
    return (X_scaled, X_mean, X_std)

#Function to scale input features (similar to sklearn robust scaler)
def robust_scale_input(X):
    X_med = np.median(X, keepdims=True)
    X_75 = np.percentile(X, 75, axis=0, keepdims=True)
    X_25 = np.percentile(X, 25, axis=0, keepdims=True)
    X_scaled = (X - X_med) / (X_75 - X_25)
    show_stats(X_scaled)
    print("Features of input array have been scaled!")
    return (X_scaled, X_med, X_75, X_25)

#Function to scale input features (limits between 0 and 1)
def max_scale_input(X):
    X_max = np.max(X, keepdims=True)
    X_scaled = (X / X_max)
    print("Features of input array have been scaled!")
    show_stats(X_scaled)
    return (X_scaled, X_max)

#Read all train images into an np array, reshape based on time-series, and delete unnecessary channels.
#X shape is (n_samples, n_time_steps, width, height, channels)
def read_train_img():
    # Train.csv has the Field_IDs needed to find the npy files
    train_temp = pd.read_csv('Train.csv')
    mask = train_temp['Quality'] != 1
    train = train_temp.loc[mask, :].copy().reindex()
    print("Adding satellite image data to Numpy array", 10*"-")
    X = np.zeros((2552, 12, 40, 40, 27))
    for idx, name in enumerate(train['Field_ID']):
        arr = np.load(f'image_arrays_train/{name}.npy')
        arr1 = arr[:,:40,:40].copy() #Resize image
        arr = arr1.reshape(12, 30, 40, 40) #Split channels from 360 to 30 channel by 12 months
        arr = np.transpose(arr, [0, 2, 3, 1]) #Push 30 channels to end (Channel Last)
        arr = np.delete(arr, [13, 14, 24], 3) #Delete empty QA10, QA20 and 'swe' bands.
        X[idx] = arr
    show_stats(X)
    y = np.asarray(train['Yield'])
    print("Shape of train array X:", X.shape)
    print("Shape of train labels y:", y.shape)
    return X,y

#Read 1055 test images
def read_test_img():
    sub = pd.read_csv('SampleSubmission.csv')
    print("Adding satellite image data to Numpy array", 10*"-")
    X = np.zeros((1055, 12, 40, 40, 27))
    for idx, name in enumerate(sub['Field_ID'].values):
        arr = np.load(f'image_arrays_test/{name}.npy')
        arr1 = arr[:,:40,:40].copy() #Resize image
        arr = arr1.reshape(12, 30, 40, 40) #Split channels from 360 to 30 channel by 12 months
        arr = np.transpose(arr, [0, 2, 3, 1]) #Push 30 channels to end (Channel Last)
        arr = np.delete(arr, [13, 14, 24], 3) #Delete empty QA10, QA20 and 'swe' bands.
        X[idx] = arr
    print("Shape of final train array:", X.shape)
    show_stats(X)
    return X

# Shuffle both input and output by indices
def shuffle(X, y):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    print("Features and labels shuffled!")
    return X_shuffled, y_shuffled

#Predict true test images and save submission file
def ready_sub(test, model, filename, batch_size=10):
    results = model.predict(test, batch_size=batch_size)
    sub = pd.read_csv('SampleSubmission.csv')
    sub['Yield'] = results
    print(sub.head())
    sub.to_csv(filename, index=False)
    print("Submission file ready and saved to ", filename,".csv")

#Create train, val, and test splits
def train_val_test_split(X, y, t):
    X_val = X[0:t]
    y_val = y[0:t]
    X_test = X[t:2*t]
    y_test = y[t:2*t]
    X_train = X[2*t:]
    y_train = y[2*t:]
    print("Train set shape: ", X_train.shape)
    print("Train target shape: ", y_train.shape)
    print("Val set shape: ", X_val.shape)
    print("Val target shape: ", y_val.shape)
    print("Test set shape: ", X_test.shape)
    print("Test target shape: ", y_test.shape)
    return X_train,y_train,X_val,y_val,X_test,y_test

#Report basic stats to ensure matrices are sclaed appropriately
def show_stats(X):
    print("Min = ", np.min(X))
    print("Max = ", np.max(X))
    print("Mean = ", np.mean(X))
    print("Std = ", np.std(X))