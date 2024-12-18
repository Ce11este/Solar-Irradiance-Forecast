import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Bidirectional, GRU, Input, Conv1D, Multiply, Flatten, MaxPooling1D, concatenate, GlobalAveragePooling1D, Reshape
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Dropout, Permute, Lambda, RepeatVector
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

df = pd.read_csv("combined_file_with_Position.csv")
df['timestamp'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
df = df.drop(columns=['Year', 'Month', 'Day', 'Hour', 'Minute'])
idx_df = df.set_index(['timestamp'])

# Data Cleaning
df2 = idx_df[['Dew Point', 'GHI', 'Clearsky GHI','Temperature','Cloud Type','Relative Humidity','Wind Speed','Wind Direction','Solar Zenith Angle','Precipitable Water','Longitude','Latitude']]
df2_cleaned = df2.dropna().query('GHI > 0')

# Normalization
max_target = df2_cleaned['GHI'].max()
min_target = df2_cleaned['GHI'].min()
x = df2_cleaned.values
x_sc = MinMaxScaler()
x = x_sc.fit_transform(x)

# Sliding window
def multivariate_data(dataset, target, start_index, end_index, history_size, target_size, step, single_step=False):
    data, labels = [], []
    start_index += history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])
        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)

train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1
total_length = len(x)
train_end_index = int(total_length * train_ratio)
val_end_index = train_end_index + int(total_length * val_ratio)
past_history, future_target, STEP = 12, 3, 1

x_train_multi, y_train_multi = multivariate_data(x, x[:, 0], 0, train_end_index, past_history, future_target, STEP, single_step=False)
x_val_multi, y_val_multi = multivariate_data(x, x[:, 0], train_end_index, val_end_index, past_history, future_target, STEP, single_step=False)
x_test_multi, y_test_multi = multivariate_data(x, x[:, 0], val_end_index, None, past_history, future_target, STEP, single_step=False)

# model construction
model = tf.keras.Sequential()
model.add(GRU(units=32, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01), input_shape=(None,12)))
model.add(Activation('relu'))
model.add(GRU(units=16, kernel_regularizer=tf.keras.regularizers.l2(0.01),return_sequences=False))
model.add(Dropout(0.5)) 
model.add(Dense(units=3)) 
model.add(Activation('relu'))
adam = Adam(learning_rate=0.001)
model.compile(optimizer=adam, loss='mean_squared_error',metrics = [tf.keras.metrics.RootMeanSquaredError()])#metrics = [tf.keras.metrics.RootMeanSquaredError()

# model training
x_train_multi_transposed = np.transpose(x_train_multi, (0, 2, 1))
checkpoint_filepath = 'Best_Model.h5'
model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
history = model.fit(x_train_multi_transposed, y_train_multi, batch_size=32, epochs=100, validation_split=0.2, shuffle=True, callbacks=[model_checkpoint_callback])

# predicting
best_model = tf.keras.models.load_model(checkpoint_filepath)
x_test_multi_transposed = np.transpose(x_test_multi, (0, 2, 1))
y_pred = best_model.predict(x_test_multi_transposed)

y_pred_unnormalized = y_pred * (max_target - min_target) + min_target
y_test_multi_unnormalized = y_test_multi * (max_target - min_target) + min_target
y_pred_original = y_pred_unnormalized
y_test_original = y_test_multi_unnormalized

# evaluation
mse = mean_squared_error(y_test_original, y_pred_original)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_original, y_pred_original)
nrmse = rmse / (np.max(y_test_original) - np.min(y_test_original))
r2 = r2_score(y_test_original, y_pred_original)
