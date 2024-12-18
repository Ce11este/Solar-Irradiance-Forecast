import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from tensorflow.keras.layers import LSTM,Bidirectional,LSTM,Input, Conv1D,Multiply,Flatten,MaxPooling1D,concatenate,GlobalAveragePooling1D,Reshape,Dense,Permute
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
import tensorflow.keras.backend as K

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

def SE_Block(input_tensor,ratio = 2):
    input_shape = K.int_shape(input_tensor)
    squeeze = tf.keras.layers.GlobalAveragePooling1D()(input_tensor)
    excitation = tf.keras.layers.Dense(units = input_shape[-1]//ratio, kernel_initializer='he_normal',activation='relu')(squeeze)
    excitation = tf.keras.layers.Dense(units = input_shape[-1],activation='sigmoid')(excitation)
    #excitation = tf.reshape(excitation, [-1, 1, input_shape[-1]])
    scale = tf.keras.layers.Multiply()([input_tensor, excitation])
    return scale

def attention_3d_block(inputs):
    input_dim = int(inputs.shape[2])
    a = inputs
    a = Dense(input_dim, activation='softmax')(a)
    # 根据给定的模式(dim)置换输入的维度  例如(2,1)即置换输入的第1和第2个维度
    a_probs = Permute((1, 2), name='attention_vec')(a)
    # Layer that multiplies (element-wise) a list of inputs.
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

def build_CNN_LSTM_model():
    visible1 = Input(shape=(12, 12))
    cnn1 = Conv1D(filters=8, kernel_size=1, activation='relu')(visible1)   #output_length = seq_length - kernel_size + 1=24-3+1=22  (22,8)
    cnn1 = MaxPooling1D(pool_size=2)(cnn1)  #output_length = ceil(seq_length / pool_size)=22/2=11   (11,8)
    cnn2 = Conv1D(filters=16, kernel_size=3, activation='relu')(cnn1)  #11-3+1=9   (9,16)
    cnn2 = MaxPooling1D(pool_size=2)(cnn2)   # (9/2,16)=(4,16)    
    cnn = Flatten()(cnn2)  
    
    visible2 = Input(shape=(12, 12))  
    lstm1 = LSTM(units=32, return_sequences=True,kernel_regularizer=tf.keras.regularizers.l2(0.01))(visible2)   #(24,32)
    lstm2 = LSTM(units=16, return_sequences=False,kernel_regularizer=tf.keras.regularizers.l2(0.01),)(lstm1)
    lstm = Flatten()(lstm2)
    
    merge = concatenate([cnn, lstm])

    attention_probs = Dense(48, activation='sigmoid', name='attention_vec')(merge)
    attention_mul = Multiply()([merge, attention_probs])
    
    
    F = Flatten()(attention_mul)
    dense = Dense(units=64, activation='relu')(F)
    output = Dense(units=3,activation='relu')(dense)
    model = Model(inputs=[visible1, visible2], outputs=output)
    adam = Adam(learning_rate=0.001)
    return model

# Correlation matrix
corr_matrix = df2_cleaned.corr().round(3)
sorted_corr = np.sort(corr_matrix.values.ravel())[::-1]
threshold = sorted_corr[int(len(sorted_corr) * 0.4)]
adj_matrix = corr_matrix.where(corr_matrix >= threshold, 0)
adj_matrix.to_csv('adjacency_matrix.csv')

adj = pd.read_csv('adjacency_matrix.csv')
adj = adj.drop(columns=[adj.columns[0]])
adj.to_csv('adjacency_matrix.csv', index=False)

model = build_CNN_LSTM_model()
model.compile(optimizer='adam', loss='mse',metrics = [tf.keras.metrics.RootMeanSquaredError()])

# model training
x_train_multi_transposed = np.transpose(x_train_multi, (0, 2, 1))
checkpoint_filepath = 'BestModel.h5'
model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=True, monitor='val_loss', mode='min', verbose=1)

history = model.fit([x_train_multi_transposed, x_train_multi_transposed], y_train_multi, batch_size=32, epochs=100, validation_split=0.2, shuffle=True, callbacks=[model_checkpoint_callback])

# predicting
best_model = tf.keras.models.load_model(checkpoint_filepath)
x_test_multi_transposed = np.transpose(x_test_multi, (0, 2, 1))
y_pred = best_model.predict([x_test_multi_transposed, x_test_multi_transposed])

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