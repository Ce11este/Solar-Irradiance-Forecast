import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Bidirectional, Input, Concatenate, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from tensorflow.keras.layers import GRU
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping
import stellargraph
from stellargraph.layer import GCN_LSTM
from stellargraph.layer import FixedAdjacencyGraphConvolution

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

class gcnModel:
    gc_layer_sizes = [16, 10]
    gc_activations = ["relu", "relu"]
    lstm_layer_sizes = [200, 200]
    lstm_activations = ["tanh", "tanh"]
    batch_size = 60

    def create_gcn_object(self, adj):
        return GCN_LSTM(
            seq_len=12,
            adj=adj,
            gc_layer_sizes=self.gc_layer_sizes,
            gc_activations=self.gc_activations,
            lstm_layer_sizes=self.lstm_layer_sizes,
            lstm_activations=self.lstm_activations,
        )

    def build_model(self, x_input, x_output, trainX, trainY, valX, valY):
        es = EarlyStopping(monitor="val_loss", patience=20)
        checkpoint_filepath = 'best_gcn_model.h5'
        mc = ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
        
        x_output = Dense(units=3, activation='linear')(x_output)  # 将输出层的 units 设置为 3

        model = Model(inputs=x_input, outputs=x_output)

        custom_learning_rate = 1e-4
        optimizer = tf.keras.optimizers.Adam(learning_rate=custom_learning_rate)
    
        model.compile(optimizer="adam", loss="mae", metrics=["mse"])
        
        history = model.fit(
            trainX,
            trainY,
            epochs=100,
            batch_size=self.batch_size,
            shuffle=True,
            validation_data=(valX, valY),
            callbacks=[es, mc]
        )
        
        return checkpoint_filepath

# Correlation matrix
corr_matrix = df2_cleaned.corr().round(3)
sorted_corr = np.sort(corr_matrix.values.ravel())[::-1]
threshold = sorted_corr[int(len(sorted_corr) * 0.4)]
adj_matrix = corr_matrix.where(corr_matrix >= threshold, 0)
adj_matrix.to_csv('adjacency_matrix.csv')

adj = pd.read_csv('adjacency_matrix.csv')
adj = adj.drop(columns=[adj.columns[0]])
adj.to_csv('adjacency_matrix.csv', index=False)

# model construction
gcn = gcnModel().create_gcn_object(adj_matrix)
x_input, x_output = gcn.in_out_tensors()

# model training
x_train_multi_transposed = np.transpose(x_train_multi, (0, 2, 1))
x_val_multi_transposed = np.transpose(x_val_multi, (0, 2, 1))
gcn_model_path = gcnModel().build_model(x_input, x_output, x_train_multi_transposed, y_train_multi, x_val_multi_transposed, y_val_multi)

gcn = gcnModel().create_gcn_object(adj_matrix)
x_input, x_output = gcn.in_out_tensors()
x_output = Dense(units=3, activation='linear')(x_output)
model = Model(inputs=x_input, outputs=x_output)

# predicting
model.load_weights(gcn_model_path)  
x_test_multi_transposed = np.transpose(x_test_multi, (0, 2, 1))
y_pred = model.predict(x_test_multi_transposed)

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
