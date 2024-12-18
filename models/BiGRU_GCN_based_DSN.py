import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, Input, Concatenate, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dropout

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

# Spatial AM
def spatial_attention(inputs, adj_matrix, K, d):
    D = K * d
    X = inputs

    query = Dense(units=D, activation='relu')(X)
    key = Dense(units=D, activation='relu')(X)
    value = Dense(units=D, activation='relu')(X)

    query = tf.concat(tf.split(query, K, axis=-1), axis=0)
    key = tf.concat(tf.split(key, K, axis=-1), axis=0)
    value = tf.concat(tf.split(value, K, axis=-1), axis=0)

    if len(query.shape) == 3:
        query = tf.expand_dims(query, axis=-2)
        key = tf.expand_dims(key, axis=-2)
        value = tf.expand_dims(value, axis=-2)
    
    query = tf.transpose(query, perm=(0, 2, 1, 3))
    key = tf.transpose(key, perm=(0, 2, 3, 1))
    value = tf.transpose(value, perm=(0, 2, 1, 3))

    attention = tf.matmul(query, key)
    attention /= (d ** 0.5)
    attention = tf.nn.softmax(attention, axis=-1)
    
    X = tf.matmul(attention, value)
    X = tf.transpose(X, perm=(0, 2, 1, 3))
    X = tf.concat(tf.split(X, K, axis=0), axis=-1)

    if len(X.shape) == 4:
        X = tf.squeeze(X, axis=-2)
    
    output = Dense(units=D, activation='relu')(X)
    return output

# Temporal AM
def temporal_attention(h_s, h_l):
    z_i_s = Dense(units=1, activation=None)(h_s)
    z_i_l = Dense(units=1, activation=None)(h_l)
    
    z_concat = tf.concat([z_i_s, z_i_l], axis=-1)
    
    w_i = tf.nn.softmax(z_concat, axis=-1)
    
    w_i_s, w_i_l = tf.split(w_i, num_or_size_splits=2, axis=-1)
    
    y_i = w_i_s * h_s + w_i_l * h_l
    
    return y_i

class NewGraphConvolution(tf.keras.layers.Layer):
    def __init__(self, output_dim, adj_matrix, dropout_rate=0.0, l2_reg=0, activation='relu', seed=1024, **kwargs):
        self.output_dim = output_dim
        self.adj_matrix = adj_matrix
        self.activation = tf.keras.activations.get(activation)
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.seed = seed
        super(NewGraphConvolution, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', shape=(input_shape[-1], self.output_dim),
                                      initializer='glorot_uniform', regularizer=tf.keras.regularizers.l2(self.l2_reg), trainable=True)
        self.bias = self.add_weight(name='bias', shape=(self.output_dim,), initializer='zeros', trainable=True)
        super(NewGraphConvolution, self).build(input_shape)

    def call(self, x):
        adj_matrix = tf.cast(self.adj_matrix, dtype=tf.float32)
        output = tf.matmul(adj_matrix, x)
        output = tf.matmul(output, self.kernel) + self.bias
        output = self.activation(output)
        return output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = super(NewGraphConvolution, self).get_config()
        config.update({
            'output_dim': self.output_dim,
            'adj_matrix': self.adj_matrix.tolist(),
            'activation': self.activation,
            'dropout_rate': self.dropout_rate,
            'l2_reg': self.l2_reg,
            'seed': self.seed
        })
        return config

    @classmethod
    def from_config(cls, config):
        config['adj_matrix'] = tf.constant(config['adj_matrix'])
        return cls(**config)

tf.keras.utils.get_custom_objects().update({'NewGraphConvolution': NewGraphConvolution})

def build_gcn_bigru_model(adj_matrix, gcn_output_dims, gru_units, input_shape, attention_heads=8, attention_dim=8):
    gcn_input = Input(shape=(input_shape[1], input_shape[2]), name='gcn_input')
    
    gcn_layer = NewGraphConvolution(output_dim=gcn_output_dims[0], adj_matrix=adj_matrix)(gcn_input)
    gcn_layer = Dense(units=gcn_output_dims[0], activation='relu')(gcn_layer)
    gcn_layer = Dropout(0.2)(gcn_layer)

    gcn_layer = NewGraphConvolution(output_dim=gcn_output_dims[1], adj_matrix=adj_matrix)(gcn_layer)
    gcn_layer = Dense(units=gcn_output_dims[1], activation='relu')(gcn_layer)
    gcn_layer = Dropout(0.2)(gcn_layer)

    gcn_layer = spatial_attention(gcn_layer, adj_matrix, attention_heads, attention_dim)
    gcn_output = tf.reduce_mean(gcn_layer, axis=1)

    gru_input = Input(shape=(input_shape[1], input_shape[2]), name='gru_input')
    bigru_layer = Bidirectional(GRU(units=gru_units, return_sequences=True))(gru_input)
    
    h_s, h_l = tf.split(bigru_layer, num_or_size_splits=2, axis=-1)
    bigru_layer = temporal_attention(h_s, h_l)
    bigru_output = tf.reduce_mean(bigru_layer, axis=1)

    combined = Concatenate()([gcn_output, bigru_output])
    dense_layer = Dense(units=32, activation='relu')(combined)
    output = Dense(units=future_target, activation='linear')(dense_layer)

    model = Model(inputs=[gcn_input, gru_input], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse', metrics=['mae'])
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

# model construction
adj_matrix = adj.values
input_shape = (None, 12, 12)
gcn_output_dims = [64, 32]
gru_units = 64
model = build_gcn_bigru_model(adj_matrix, gcn_output_dims, gru_units, input_shape)

# model training
x_train_multi_transposed = np.transpose(x_train_multi, (0, 2, 1))
checkpoint_filepath = 'BestModel.h5'
model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=True, monitor='val_loss', mode='min', verbose=1)

history = model.fit([x_train_multi_transposed, x_train_multi_transposed], y_train_multi, batch_size=32, epochs=100, validation_split=0.2, shuffle=True, callbacks=[model_checkpoint_callback])

# predicting
best_model = tf.keras.models.load_model(checkpoint_filepath, custom_objects={'NewGraphConvolution': NewGraphConvolution})
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
