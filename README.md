# training-time-predictor  
  
This repository intends to provide a powerful toolbox for your machine learning model training time prediction, which includes training time data generation, training time pre-trained models for both tensorflow and pytorch environment.   
  
Currently, this repository is still under developing.

## <div align="center">Documentation</div>  
See the [Research Docs ](https://docs.google.com/document/d/1FLgQ58umOK8FmGb_iNfiACIAdGIlFEFSvfOBEQXfsyY/edit) for full documentation on motivation, related researches.
## <div align="center">GPU Environment Check </div>  
<details open>  
Model training times depending on three factors, environment, model structure and data. Environment Check should be the first check before generating any data point. 

Environment configuration should be features for training time prediction for your own experiment if you have multiple environments.

Reference [**here**](https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries) for all feature names.
```python
from env_detect import gpu_features
# need to run with GPU  

# only show default features  
gpufeature = gpu_features()  

>>>print(gpufeature.get_features())
{'timestamp': '2022/05/27 17:39:01.851', 'driver_version': '460.73.01', 'count': '1', 'name': 'Tesla T4', 'pcie.link.width.max': '16', 'vbios_version': '90.04.96.00.01', 'memory.total [MiB]': '15109 MiB', 'temperature.gpu': '46'}
```

```python
# with additional features  
gpufeature = gpu_features(  
    features=['power.management', 'power.limit'], with_dafault_features=True  
)  

>>>print(gpufeature.get_features())
{'timestamp': '2022/05/27 17:39:01.896', 'driver_version': '460.73.01', 'count': '1', 'name': 'Tesla T4', 'pcie.link.width.max': '16', 'vbios_version': '90.04.96.00.01', 'memory.total [MiB]': '15109 MiB', 'temperature.gpu': '46', 'power.management': 'Enabled', 'power.limit [W]': '70.00 W'}
```

```python
# only use wanted features  
gpufeature = gpu_features(  
    features=['power.management', 'power.limit'], with_dafault_features=False  
)  

>>>print(gpufeature.get_features())
{'power.management': 'Enabled', 'power.limit [W]': '70.00 W'}
```

</details>  


## <div align="center">Data Generation Quick Start Examples</div>  
<details open>  
<summary>Install</summary>  
  
Clone repo and install [requirements.txt](https://github.com/aipaca-mlops/ML-training-cost-predictor/blob/master/requirements.txt) in a  [**Python>=3.7.0**](https://www.python.org/) environment, including  [**Tensorflow>=2.3**](https://www.tensorflow.org/versions), including  [**Keras>=2.8.0**](https://github.com/keras-team/keras/releases).  

```bash  
git clone https://github.com/aipaca-mlops/ML-training-cost-predictor.git  # clone 
cd ML-training-cost-calculator 
pip install -r requirements.txt  # install
```  
</details>  

<details open>  
<summary>Generate Dense Model</summary>  

Generate model configs for feed forward network.

```python  
import random  
import matplotlib.pyplot as plt  
from model_level_utils import gen_nn, model_train_data
  
# generate model configurations as data points  
data_points = 1000  
gnn = gen_nn(  
			hidden_layers_num_lower=1, #lower bound for layer size
			hidden_layers_num_upper=51, #upper bound for layer size
			hidden_layer_size_lower=1, #lower bound for layer number
			hidden_layer_size_upper=1001, #upper bound for layer number
			activation='random',  
			optimizer='random',  
			loss='random'  
			)  
model_configs = gnn.generate_model_configs(num_model_data=data_points)  
```  
Check out what is inside model_configs.
```python  
>>>print(type(model_configs))
<class 'list'>

>>>print(model_configs[0])
{'layer_sizes': [966, 624, 193], 
'activations': ['relu', 'selu', 'exponential'], 
'optimizer': 'adagrad', 
'loss': 'poisson'}
```
Next we can use generated model configurations to get training times. This might take a while depending on your GPU.
```python  
# train generated model configurations to get training time  
mtd = model_train_data(  
			model_configs,  
			input_dims=list(range(1, 1001)),  
			batch_sizes=[2**i for i in range(1, 9)],  
			epochs=5,  
			truncate_from=1,  
			trials=2,  
			batch_strategy='random',  
			)  
model_data = mtd.get_train_data()
```
Check out what is inside model_data.
```python  
>>>print(type(model_data))
<class 'list'>

>>>print(model_data[0])
{'layer_sizes': [966, 624, 193], 
'activations': ['relu', 'selu', 'exponential'], 
'optimizer': 'adagrad', 
'loss': 'poisson', 
'batch_size_8': {'batch_time': 1.8298625946044922, #medium training time for one batch
				'epoch_time': 2.166271209716797, #medium training time for one epoch
				'setup_time': 449.7561454772949, #tensorflow build graph time
				'input_dim': 196}
				}
```
Now we can convert generated data into pandas dataframe.
```python  
# convert raw data as dataframe and scaler  
df, scaler = mtd.convert_config_data(  
    model_data, layer_num_upper=50, layer_na_fill=0, act_na_fill=0, min_max_scaler=True  
)
```
```python  
>>>display(df.head())
|    |   layer_1_size |   layer_2_size |   layer_3_size |   layer_4_size |   layer_5_size |   layer_6_size |   layer_7_size |   layer_8_size |   layer_9_size |   layer_10_size |   layer_11_size |   layer_12_size |   layer_13_size |   layer_14_size |   layer_15_size |   layer_16_size |   layer_17_size |   layer_18_size |   layer_19_size |   layer_20_size |   layer_21_size |   layer_22_size |   layer_23_size |   layer_24_size |   layer_25_size |   layer_26_size |   layer_27_size |   layer_28_size |   layer_29_size |   layer_30_size |   layer_31_size |   layer_32_size |   layer_33_size |   layer_34_size |   layer_35_size |   layer_36_size |   layer_37_size |   layer_38_size |   layer_39_size |   layer_40_size |   layer_41_size |   layer_42_size |   layer_43_size |   layer_44_size |   layer_45_size |   layer_46_size |   layer_47_size |   layer_48_size |   layer_49_size |   layer_50_size |   layer_1_activation |   layer_2_activation |   layer_3_activation |   layer_4_activation |   layer_5_activation |   layer_6_activation |   layer_7_activation |   layer_8_activation |   layer_9_activation |   layer_10_activation |   layer_11_activation |   layer_12_activation |   layer_13_activation |   layer_14_activation |   layer_15_activation |   layer_16_activation |   layer_17_activation |   layer_18_activation |   layer_19_activation |   layer_20_activation |   layer_21_activation |   layer_22_activation |   layer_23_activation |   layer_24_activation |   layer_25_activation |   layer_26_activation |   layer_27_activation |   layer_28_activation |   layer_29_activation |   layer_30_activation |   layer_31_activation |   layer_32_activation |   layer_33_activation |   layer_34_activation |   layer_35_activation |   layer_36_activation |   layer_37_activation |   layer_38_activation |   layer_39_activation |   layer_40_activation |   layer_41_activation |   layer_42_activation |   layer_43_activation |   layer_44_activation |   layer_45_activation |   layer_46_activation |   layer_47_activation |   layer_48_activation |   layer_49_activation |   layer_50_activation |   batch_size |   input_dim |   optimizer_adadelta |   optimizer_adagrad |   optimizer_adam |   optimizer_adamax |   optimizer_ftrl |   optimizer_nadam |   optimizer_rmsprop |   optimizer_sgd |   loss_categorical_crossentropy |   loss_mae |   loss_mape |   loss_mse |   loss_msle |   loss_poisson |   batch_time |   epoch_time |   setup_time |
|---:|---------------:|---------------:|---------------:|---------------:|---------------:|---------------:|---------------:|---------------:|---------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|---------------------:|---------------------:|---------------------:|---------------------:|---------------------:|---------------------:|---------------------:|---------------------:|---------------------:|----------------------:|----------------------:|----------------------:|----------------------:|----------------------:|----------------------:|----------------------:|----------------------:|----------------------:|----------------------:|----------------------:|----------------------:|----------------------:|----------------------:|----------------------:|----------------------:|----------------------:|----------------------:|----------------------:|----------------------:|----------------------:|----------------------:|----------------------:|----------------------:|----------------------:|----------------------:|----------------------:|----------------------:|----------------------:|----------------------:|----------------------:|----------------------:|----------------------:|----------------------:|----------------------:|----------------------:|----------------------:|----------------------:|----------------------:|----------------------:|----------------------:|-------------:|------------:|---------------------:|--------------------:|-----------------:|-------------------:|-----------------:|------------------:|--------------------:|----------------:|--------------------------------:|-----------:|------------:|-----------:|------------:|---------------:|-------------:|-------------:|-------------:|
|  0 |       0.54509  |       0.202608 |       0.823824 |      0.468468  |          0.348 |          0.591 |          0.102 |      0.0460922 |       0.770771 |        0.721163 |           0.608 |        0.304304 |        0.956871 |           0.172 |           0.403 |        0.306613 |        0.640281 |           0.471 |        0.108108 |       0.956871  |        0.94995  |        0.413413 |        0.384384 |           0.126 |        0.782347 |        0.536537 |        0.744745 |       0.531595  |           0.544 |        0.947896 |        0.50303  |           0.868 |        0.155936 |        0.229229 |           0.932 |        0.951904 |        0.276104 |        0.963855 |        0.611836 |        0.291291 |        0.284422 |      0.00902708 |               0 |               0 |               0 |               0 |               0 |               0 |               0 |               0 |                0.875 |             0.333333 |             0.111111 |             0.333333 |             0.222222 |             0.777778 |             0.444444 |             0.888889 |             0.444444 |              0.888889 |              0.444444 |              0.111111 |              0.777778 |              0.222222 |              0.333333 |              0.444444 |              0.666667 |              0.777778 |              1        |              0.222222 |              0.888889 |              0.666667 |              0.222222 |              0.555556 |              0.444444 |              0.555556 |              1        |              0.444444 |              0.888889 |              1        |              0.111111 |              0.222222 |              0.111111 |              0.888889 |              0.222222 |              0.777778 |              0.444444 |              0.444444 |              1        |              0.555556 |              0.222222 |              0.777778 |                     0 |                     0 |                     0 |                     0 |                     0 |                     0 |                     0 |                     0 |    0.496063  |   0.534068  |                    0 |                   0 |                1 |                  0 |                0 |                 0 |                   0 |               0 |                               0 |          0 |           1 |          0 |           0 |              0 |     12.0096  |     12.3415  |     5442.04  |
|  1 |       0.966934 |       0.625878 |       0.193193 |      0         |          0     |          0     |          0     |      0         |       0        |        0        |           0     |        0        |        0        |           0     |           0     |        0        |        0        |           0     |        0        |       0         |        0        |        0        |        0        |           0     |        0        |        0        |        0        |       0         |           0     |        0        |        0        |           0     |        0        |        0        |           0     |        0        |        0        |        0        |        0        |        0        |        0        |      0          |               0 |               0 |               0 |               0 |               0 |               0 |               0 |               0 |                0     |             0.777778 |             1        |             0        |             0        |             0        |             0        |             0        |             0        |              0        |              0        |              0        |              0        |              0        |              0        |              0        |              0        |              0        |              0        |              0        |              0        |              0        |              0        |              0        |              0        |              0        |              0        |              0        |              0        |              0        |              0        |              0        |              0        |              0        |              0        |              0        |              0        |              0        |              0        |              0        |              0        |              0        |                     0 |                     0 |                     0 |                     0 |                     0 |                     0 |                     0 |                     0 |    0.023622  |   0.194389  |                    0 |                   1 |                0 |                  0 |                0 |                 0 |                   0 |               0 |                               0 |          0 |           0 |          0 |           0 |              1 |      1.82986 |      2.16627 |      449.756 |
|  2 |       0.433868 |       0.212638 |       0.744745 |      0.0970971 |          0.473 |          0.67  |          0.109 |      0.845691  |       0.49049  |        0.375125 |           0.41  |        0.854855 |        0.88666  |           0.336 |           0.211 |        0.5501   |        0.354709 |           0.484 |        0.725726 |       0.766299  |        0.577578 |        0.119119 |        0.376376 |           0.288 |        0.822467 |        0.814815 |        0.570571 |       0.438315  |           0.241 |        0.477956 |        0.780808 |           0.943 |        0.151911 |        0.251251 |           0     |        0        |        0        |        0        |        0        |        0        |        0        |      0          |               0 |               0 |               0 |               0 |               0 |               0 |               0 |               0 |                0     |             0.888889 |             0.555556 |             0.777778 |             0.111111 |             0.777778 |             0.444444 |             0.555556 |             0.333333 |              0.888889 |              0.333333 |              0.777778 |              0.555556 |              0.333333 |              0.777778 |              0.222222 |              0.444444 |              0.444444 |              0.777778 |              0.888889 |              0.555556 |              0.333333 |              0.888889 |              0.555556 |              0.555556 |              0.777778 |              0.222222 |              0.444444 |              0.555556 |              0.888889 |              1        |              0.555556 |              0.222222 |              0.444444 |              0        |              0        |              0        |              0        |              0        |              0        |              0        |              0        |                     0 |                     0 |                     0 |                     0 |                     0 |                     0 |                     0 |                     0 |    0.244094  |   0.837675  |                    0 |                   0 |                0 |                  0 |                0 |                 1 |                   0 |               0 |                               0 |          0 |           0 |          1 |           0 |              0 |     18.2607  |     18.6019  |     4921.17  |
|  3 |       0.562124 |       0.487462 |       0.626627 |      0.275275  |          0.631 |          0.817 |          0.875 |      0.383768  |       0.225225 |        0.842528 |           0.509 |        0.477477 |        0.131394 |           0.303 |           0.066 |        0.641283 |        0.845691 |           0.633 |        0.295295 |       0.415246  |        0.8999   |        0.655656 |        0.351351 |           0.566 |        0.489468 |        0.492492 |        0.947948 |       0.0802407 |           0.253 |        0.140281 |        0.458586 |           0.598 |        0.587525 |        0.274274 |           0.085 |        0.96493  |        0.172691 |        0.348394 |        0.805416 |        0.745746 |        0.954774 |      0.907723   |               0 |               0 |               0 |               0 |               0 |               0 |               0 |               0 |                0.25  |             0.555556 |             0.111111 |             0.555556 |             0.333333 |             0.555556 |             0.111111 |             1        |             0.222222 |              0.222222 |              0.888889 |              0.444444 |              1        |              0.111111 |              0.444444 |              0.222222 |              0.888889 |              0.777778 |              0.777778 |              0.333333 |              0.888889 |              0.222222 |              0.777778 |              1        |              0.555556 |              0.888889 |              0.666667 |              0.555556 |              0.666667 |              0.777778 |              0.888889 |              0.333333 |              0.888889 |              1        |              0.333333 |              0.222222 |              0.888889 |              0.111111 |              0.777778 |              0.888889 |              1        |              0.111111 |                     0 |                     0 |                     0 |                     0 |                     0 |                     0 |                     0 |                     0 |    0.0551181 |   0.311623  |                    0 |                   1 |                0 |                  0 |                0 |                 0 |                   0 |               0 |                               0 |          1 |           0 |          0 |           0 |              0 |      8.15439 |      8.47578 |     1590.8   |
|  4 |       0.893788 |       0.11334  |       0.148148 |      0.947948  |          0.858 |          0.797 |          0.268 |      0.833667  |       0.575576 |        0.721163 |           0.607 |        0.001001 |        0.834504 |           0.133 |           0.064 |        0.169339 |        0.655311 |           0.206 |        0.913914 |       0.0832497 |        0.36036  |        0.51952  |        0.331331 |           0.279 |        0.2668   |        0.4004   |        0.301301 |       0.728185  |           0.4   |        0.891784 |        0.240404 |           0.474 |        0.349095 |        0.91992  |           0.898 |        0.163327 |        0        |        0        |        0        |        0        |        0        |      0          |               0 |               0 |               0 |               0 |               0 |               0 |               0 |               0 |                0.5   |             0.333333 |             0.333333 |             1        |             0.111111 |             0.111111 |             1        |             0.222222 |             1        |              1        |              0.777778 |              0.888889 |              0.666667 |              0.888889 |              0.222222 |              0.777778 |              0.666667 |              0.333333 |              0.111111 |              0.444444 |              0.555556 |              0.444444 |              0.222222 |              0.777778 |              0.888889 |              0.777778 |              0.111111 |              1        |              1        |              0.777778 |              0.777778 |              0.444444 |              0.777778 |              0.555556 |              0.222222 |              0.444444 |              0        |              0        |              0        |              0        |              0        |              0        |                     0 |                     0 |                     0 |                     0 |                     0 |                     0 |                     0 |                     0 |    0.0551181 |   0.0751503 |                    0 |                   1 |                0 |                  0 |                0 |                 0 |                   0 |               0 |                               0 |          0 |           1 |          0 |           0 |              0 |      6.92463 |      7.37071 |     1709.9   |
```

</details>  


<details open>  
<summary>Generate CNN Model</summary>  

Generate model configs for convolutional network.

```python  
from model_trainingtime_prediction.model_level_utils_cnn import gen_cnn2d, cnn2d_model_train_data
import nltk
from tqdm import tqdm

gen = gen_cnn2d(  
		input_shape_lower=20,  
		input_shape_upper=101,  
		conv_layer_num_lower=1,  
		conv_layer_num_upper=51,  
		filter_lower=1,  
		filter_upper=101,  
		dense_layer_num_lower=1,  
		dense_layer_num_upper=6,  
		dense_size_lower=1,  
		dense_size_upper=1001,  
		max_pooling_prob=.5,  
		input_channels=None,  
		paddings=None,  
		activations=None,  
		optimizers=None,  
		losses=None  
		)  
model_configs = gen.generate_model_configs(num_model_data=data_points, progress=True)
```  
Check out what is inside model_configs.

```python  
>>>print(type(model_configs))
<class 'list'>

>>>print(model_configs[0])
[[{'filters': 83, 'padding': 'valid', 'activation': 'softmax', 'kernel_size': (53, 53), 'strides': (1, 1)}, 
{'filters': 51, 'padding': 'same', 'activation': 'softsign', 'kernel_size': (2, 2), 'strides': (1, 1)}, 
{'padding': 'same', 'pool_size': (2, 2), 'strides': (1, 1)}, 
{'filters': 17, 'padding': 'same', 'activation': 'selu', 'kernel_size': (3, 3), 'strides': (1, 1)}, 
{'padding': 'same', 'pool_size': (3, 3), 'strides': (1, 1)}, 
{'filters': 32, 'padding': 'valid', 'activation': 'tanh', 'kernel_size': (2, 2), 'strides': (2, 2)}, 
{'padding': 'valid', 'pool_size': (1, 1), 'strides': (1, 1)}, 
{}, 
{'units': 702, 'activation': 'sigmoid'}, 
{'units': 471, 'activation': 'elu'}, 
{'units': 906, 'activation': 'elu'}, 
{'Compile': {'optimizer': 'rmsprop', 'loss': 'categorical_crossentropy'}, 'Fit': {}}], 
['Conv2D', 'Conv2D', 'MaxPooling2D', 'Conv2D', 'MaxPooling2D', 'Conv2D', 'MaxPooling2D', 'Flatten', 'Dense', 'Dense', 'Dense'], 
(56, 56, 3)]

```
Next we can use generated model configurations to get training times. This might take a while depending on your GPU.
```python  
# train generated model configurations to get training time  
mtd = cnn2d_model_train_data(
	model_configs, batch_sizes=None, epochs=None, truncate_from=None, trials=None
)

model_data = mtd.get_train_data(progress=True)
```
Check out what is inside model_data.
```python  
>>>print(type(model_data))
<class 'list'>

>>>print(model_data[0])
[[{'filters': 83, 'padding': 'valid', 'activation': 'softmax', 'kernel_size': (53, 53), 'strides': (1, 1)}, 
{'filters': 51, 'padding': 'same', 'activation': 'softsign', 'kernel_size': (2, 2), 'strides': (1, 1)}, 
{'padding': 'same', 'pool_size': (2, 2), 'strides': (1, 1)}, 
{'filters': 17, 'padding': 'same', 'activation': 'selu', 'kernel_size': (3, 3), 'strides': (1, 1)}, 
{'padding': 'same', 'pool_size': (3, 3), 'strides': (1, 1)}, 
{'filters': 32, 'padding': 'valid', 'activation': 'tanh', 'kernel_size': (2, 2), 'strides': (2, 2)}, 
{'padding': 'valid', 'pool_size': (1, 1), 'strides': (1, 1)}, 
{}, 
{'units': 702, 'activation': 'sigmoid'}, 
{'units': 471, 'activation': 'elu'}, 
{'units': 906, 'activation': 'elu'}, 
{'Compile': {'optimizer': 'rmsprop', 'loss': 'categorical_crossentropy'}, 'Fit': {}}], 
['Conv2D', 'Conv2D', 'MaxPooling2D', 'Conv2D', 'MaxPooling2D', 'Conv2D', 'MaxPooling2D', 'Flatten', 'Dense', 'Dense', 'Dense'], 
(56, 56, 3), 
{'batch_size': 4, 
'batch_time': 3.8396120071411133, 
'epoch_time': 4.180788993835449, 
'setup_time': 930.9759140014648, 
'input_dim': (56, 56, 3)}]

```
Now we can convert generated data into pandas dataframes.
```python  
# 15 from conv_layer_num_upper * 2 + dense_layer_num_upper
# * 2 because the maxpooling layer might be there

model_data_dfs, time_df, scaler = mtd.convert_config_data(
	model_data, max_layer_num=15, num_fill_na=0, name_fill_na=None, min_max_scaler=True
)
```
```python  
>>>display(model_data_dfs[0])
|    |   layer_size |   kernel_size |   strides |   batch_size |   input_shape |   channels |   layer_type_Conv2D |   layer_type_Dense |   layer_type_MaxPooling2D |   padding_same |   padding_valid |   activation_elu |   activation_exponential |   activation_relu |   activation_selu |   activation_sigmoid |   activation_softmax |   activation_softplus |   activation_softsign |   activation_tanh |   optimizer_adadelta |   optimizer_adagrad |   optimizer_adam |   optimizer_adamax |   optimizer_ftrl |   optimizer_nadam |   optimizer_rmsprop |   optimizer_sgd |   loss_categorical_crossentropy |   loss_mae |   loss_mape |   loss_mse |   loss_msle |   loss_poisson |
|---:|-------------:|--------------:|----------:|-------------:|--------------:|-----------:|--------------------:|-------------------:|--------------------------:|---------------:|----------------:|-----------------:|-------------------------:|------------------:|------------------:|---------------------:|---------------------:|----------------------:|----------------------:|------------------:|---------------------:|--------------------:|-----------------:|-------------------:|-----------------:|------------------:|--------------------:|----------------:|--------------------------------:|-----------:|------------:|-----------:|------------:|---------------:|
|  0 |        0.083 |     0.557895  |       0.5 |     0.015625 |      0.565657 |          1 |                   1 |                  0 |                         0 |              0 |               1 |                0 |                        0 |                 0 |                 0 |                    0 |                    1 |                     0 |                     0 |                 0 |                    0 |                   0 |                0 |                  0 |                0 |                 0 |                   1 |               0 |                               1 |          0 |           0 |          0 |           0 |              0 |
|  1 |        0.051 |     0.0210526 |       0.5 |     0.015625 |      0.565657 |          1 |                   1 |                  0 |                         0 |              1 |               0 |                0 |                        0 |                 0 |                 0 |                    0 |                    0 |                     0 |                     1 |                 0 |                    0 |                   0 |                0 |                  0 |                0 |                 0 |                   1 |               0 |                               1 |          0 |           0 |          0 |           0 |              0 |
|  2 |        0     |     0.0210526 |       0.5 |     0.015625 |      0.565657 |          1 |                   0 |                  0 |                         1 |              1 |               0 |                0 |                        0 |                 0 |                 0 |                    0 |                    0 |                     0 |                     0 |                 0 |                    0 |                   0 |                0 |                  0 |                0 |                 0 |                   1 |               0 |                               1 |          0 |           0 |          0 |           0 |              0 |
|  3 |        0.017 |     0.0315789 |       0.5 |     0.015625 |      0.565657 |          1 |                   1 |                  0 |                         0 |              1 |               0 |                0 |                        0 |                 0 |                 1 |                    0 |                    0 |                     0 |                     0 |                 0 |                    0 |                   0 |                0 |                  0 |                0 |                 0 |                   1 |               0 |                               1 |          0 |           0 |          0 |           0 |              0 |
|  4 |        0     |     0.0315789 |       0.5 |     0.015625 |      0.565657 |          1 |                   0 |                  0 |                         1 |              1 |               0 |                0 |                        0 |                 0 |                 0 |                    0 |                    0 |                     0 |                     0 |                 0 |                    0 |                   0 |                0 |                  0 |                0 |                 0 |                   1 |               0 |                               1 |          0 |           0 |          0 |           0 |              0 |
|  5 |        0.032 |     0.0210526 |       1   |     0.015625 |      0.565657 |          1 |                   1 |                  0 |                         0 |              0 |               1 |                0 |                        0 |                 0 |                 0 |                    0 |                    0 |                     0 |                     0 |                 1 |                    0 |                   0 |                0 |                  0 |                0 |                 0 |                   1 |               0 |                               1 |          0 |           0 |          0 |           0 |              0 |
|  6 |        0     |     0.0105263 |       0.5 |     0.015625 |      0.565657 |          1 |                   0 |                  0 |                         1 |              0 |               1 |                0 |                        0 |                 0 |                 0 |                    0 |                    0 |                     0 |                     0 |                 0 |                    0 |                   0 |                0 |                  0 |                0 |                 0 |                   1 |               0 |                               1 |          0 |           0 |          0 |           0 |              0 |
|  7 |        0.702 |     0         |       0   |     0.015625 |      0.565657 |          1 |                   0 |                  1 |                         0 |              0 |               0 |                0 |                        0 |                 0 |                 0 |                    1 |                    0 |                     0 |                     0 |                 0 |                    0 |                   0 |                0 |                  0 |                0 |                 0 |                   1 |               0 |                               1 |          0 |           0 |          0 |           0 |              0 |
|  8 |        0.471 |     0         |       0   |     0.015625 |      0.565657 |          1 |                   0 |                  1 |                         0 |              0 |               0 |                1 |                        0 |                 0 |                 0 |                    0 |                    0 |                     0 |                     0 |                 0 |                    0 |                   0 |                0 |                  0 |                0 |                 0 |                   1 |               0 |                               1 |          0 |           0 |          0 |           0 |              0 |
|  9 |        0.906 |     0         |       0   |     0.015625 |      0.565657 |          1 |                   0 |                  1 |                         0 |              0 |               0 |                1 |                        0 |                 0 |                 0 |                    0 |                    0 |                     0 |                     0 |                 0 |                    0 |                   0 |                0 |                  0 |                0 |                 0 |                   1 |               0 |                               1 |          0 |           0 |          0 |           0 |              0 |
| 10 |        0     |     0         |       0   |     0        |      0        |          0 |                   0 |                  0 |                         0 |              0 |               0 |                0 |                        0 |                 0 |                 0 |                    0 |                    0 |                     0 |                     0 |                 0 |                    0 |                   0 |                0 |                  0 |                0 |                 0 |                   0 |               0 |                               0 |          0 |           0 |          0 |           0 |              0 |
| 11 |        0     |     0         |       0   |     0        |      0        |          0 |                   0 |                  0 |                         0 |              0 |               0 |                0 |                        0 |                 0 |                 0 |                    0 |                    0 |                     0 |                     0 |                 0 |                    0 |                   0 |                0 |                  0 |                0 |                 0 |                   0 |               0 |                               0 |          0 |           0 |          0 |           0 |              0 |
| 12 |        0     |     0         |       0   |     0        |      0        |          0 |                   0 |                  0 |                         0 |              0 |               0 |                0 |                        0 |                 0 |                 0 |                    0 |                    0 |                     0 |                     0 |                 0 |                    0 |                   0 |                0 |                  0 |                0 |                 0 |                   0 |               0 |                               0 |          0 |           0 |          0 |           0 |              0 |
| 13 |        0     |     0         |       0   |     0        |      0        |          0 |                   0 |                  0 |                         0 |              0 |               0 |                0 |                        0 |                 0 |                 0 |                    0 |                    0 |                     0 |                     0 |                 0 |                    0 |                   0 |                0 |                  0 |                0 |                 0 |                   0 |               0 |                               0 |          0 |           0 |          0 |           0 |              0 |
| 14 |        0     |     0         |       0   |     0        |      0        |          0 |                   0 |                  0 |                         0 |              0 |               0 |                0 |                        0 |                 0 |                 0 |                    0 |                    0 |                     0 |                     0 |                 0 |                    0 |                   0 |                0 |                  0 |                0 |                 0 |                   0 |               0 |                               0 |          0 |           0 |          0 |           0 |              0 |
```
```python  
>>>display(time_df.head())
|    |   batch_time |   epoch_time |   setup_time |
|---:|-------------:|-------------:|-------------:|
|  0 |      3.83961 |      4.18079 |      930.976 |
|  1 |     13.6241  |     13.9616  |      405.777 |
|  2 |    152.279   |    152.747   |    10592.4   |
|  3 |    364.004   |    364.5     |    26533.8   |
|  4 |     30.2401  |     30.6295  |     1840.11  |
```
</details>  

<details open>  
<summary>Generate Classic CNN Model Data</summary>  

People normally use pre-defined classic CNN structures instead of creating their own. We can also generate data for these specific models instead of random generated CNN models. This will help us increase the prediction accuracy.

For valid pre-defined model names check [**here**](https://keras.io/api/applications/).

For each pre-trained models, the input shape cannot be changed if we want to fine-tune on pre-trained weights.

```python
from model_level_utils_cnn import ClassicModelTrainData

cmtd = ClassicModelTrainData(
		batch_sizes=[2, 4],
		optimizers=["sgd", "rmsprop", "adam"],
		losses=["mse", "msle", "poisson", "categorical_crossentropy"])

model_data = cmtd.get_train_data('VGG16', output_size=1000, progress=True)
```  
```python
batch_sizes = [i['batch_size'] for i in model_data]
optimizers = [i['optimizer'] for i in model_data]
losses = [i['loss'] for i in model_data]
times = [i['batch_time'] for i in model_data]
```
```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# set plot style: grey grid in the background:
sns.set(style="darkgrid")

data = pd.DataFrame(list(zip(batch_sizes, optimizers, losses, times)), columns = ['batch_size', 'optimizer', 'loss', 'batch_time'])

sns.set(font_scale = 2)
ax = sns.catplot(x="optimizer", y="batch_time",
             hue="loss", col="batch_size",
             data=data, kind="bar",
             height=10, aspect=1)
plt.show()
```
![enter image description here](https://raw.github.com/aipaca-mlops/ML-training-cost-calculator/create_readme_xin/Images/DenseRegressionHistory.png)

It is interesting to observe that loss functions play less important role in terms of training time spending. Whereas batch sizes and optimizers have much apparent impact as we expected. 
</details>  



<details open>  
<summary>Generate RNN Model</summary>  

Generate training time data for recurrent network.

```python  
put python steps here
```  
</details>  

## <div align="center">Training Time Prediction Quick Start Examples</div>  

<details open>  
<summary>Predict Dense Model Training Time Using Model Level Structure</summary>  

Make a prediction for a feed forward network training time. 

Build a regression model with generated dataframe data from **Dense Model Data Generation**.
```python  
import numpy as np

# use data to train a ML model  
test_ratio = 0.2  
df_index = df.index.tolist()  
np.random.shuffle(df_index)  
  
middle_index = int(df.shape[0] * test_ratio)  
test_idx = df_index[:middle_index]  
train_idx = df_index[middle_index:]  
  
df_train = df.iloc[train_idx]  
df_test = df.iloc[test_idx]  
  
# we need to train 2 models, one to predict batch runtime, one to predict setup time  
# combine both will be the true training time of a model  
feature_cols = df.columns.tolist()[:-3] #last 3 columns are target columns
target_col = 'batch_time'  
setup_col = 'setup_time'  
  
x_train = df_train[feature_cols].to_numpy()  
y_batch_train = np.array(df_train[target_col].tolist())  
y_setup_train = np.array(df_train[setup_col].tolist())  
  
x_test = df_test[feature_cols].to_numpy()  
y_batch_test = np.array(df_test[target_col].tolist())  
y_setup_test = np.array(df_test[setup_col].tolist())  
  
# build a regression dense model for batch time prediction  
from keras.models import Sequential  
from keras.layers import Dense  
  
batch_model = Sequential()  
batch_model.add(  
    Dense(200, input_dim=x_train.shape[1], kernel_initializer='normal', activation='relu')  
)  
batch_model.add(Dense(200, kernel_initializer='normal', activation='relu'))  
batch_model.add(Dense(200, kernel_initializer='normal', activation='relu'))  
batch_model.add(Dense(200, kernel_initializer='normal', activation='relu'))  
batch_model.add(Dense(1, kernel_initializer='normal'))  
# Compile model  
batch_model.compile(loss='mean_squared_error', optimizer='adam')  
  
history_batch = batch_model.fit(  
	x_train,  
	y_batch_train,  
	batch_size=16,  
	epochs=50,  
	validation_data=(x_test, y_batch_test),  
	verbose=True  
)
```

Plot training history.
```python  
# summarize history for loss
plt.plot(history_batch.history['loss'])
plt.plot(history_batch.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train',  'test'], loc='upper left')
plt.show()
```  
![enter image description here](https://raw.github.com/aipaca-mlops/ML-training-cost-calculator/create_readme_xin/Images/DenseRegressionHistory.png)

Plot prediction vs test.
```python  
# plot predictions vs true for batch model
batch_y_pred = batch_model.predict(x_test)
batch_y_pred = batch_y_pred.reshape(batch_y_pred.shape[0],  )
plt.scatter(batch_y_pred, y_batch_test)
plt.show()
```  
![enter image description here](https://raw.github.com/aipaca-mlops/ML-training-cost-calculator/create_readme_xin/Images/DensePredVSTest.png)

Now we train regression model for setup time model.
```python  
# build a dense model for setup time prediction  
setup_model = Sequential()  
setup_model.add(  
    Dense(200, input_dim=x_train.shape[1], kernel_initializer='normal', activation='relu')  
)  
setup_model.add(Dense(200, kernel_initializer='normal', activation='relu'))  
setup_model.add(Dense(200, kernel_initializer='normal', activation='relu'))  
setup_model.add(Dense(200, kernel_initializer='normal', activation='relu'))  
setup_model.add(Dense(1, kernel_initializer='normal'))  
# Compile model  
setup_model.compile(loss='mean_squared_error', optimizer='adam')  
history_setup = setup_model.fit(  
	x_train,  
	y_setup_train,  
	batch_size=16,  
	epochs=45,  
	validation_data=(x_test, y_setup_test),  
	verbose=True  
)
```  

Plot training history.
```python  
# summarize history for loss
plt.plot(history_setup.history['loss'])
plt.plot(history_setup.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train',  'test'], loc='upper left')
plt.show()
```  
![enter image description here](https://raw.github.com/aipaca-mlops/ML-training-cost-calculator/create_readme_xin/Images/DenseSetuptimeHistory.png)

Plot prediction vs test.
```python  
# plot predictions vs true for setup time model
setup_y_pred = setup_model.predict(x_test)
setup_y_pred = setup_y_pred.reshape(setup_y_pred.shape[0],  )
plt.scatter(setup_y_pred, y_setup_test)
plt.show()
```  
![enter image description here](https://raw.github.com/aipaca-mlops/ML-training-cost-calculator/create_readme_xin/Images/DenseSetuptimePredVSTest.png)
</details>  

<details open>  
<summary>Predict Dense Model Training Time Using Unit 
Counts</summary>  

Now we do not convert model structure into features, instead, we simply use the sum of all units of all dense layers as the model feature. It turns out works great and gives much more flexibility.

```python
from model_level_utils import convert_dense_data

cdd = convert_dense_data()
# model_data from data generation step
dense_data, times_data, Scaler = cdd.convert_model_config(model_data, data_type='Units', min_max_scaler=True)
```

Take of look of dense_data and times_data

```python
>>>print(dense_data)
[[0.20932152 0.05511811 0. ... 0. 0. 0. ] 
 [0.90319687 0.11811024 0. ... 0. 1. 0. ] 
 [0.68745291 0.02362205 0. ... 0. 0. 0. ] 
 ... 
 [0.54267877 0.05511811 0. ... 0. 0. 1. ] 
 [0.09124179 0. 0. ... 0. 0. 0. ] 
 [0.46047863 1. 0. ... 1. 0. 0. ]]
 
>>>print(times_data)
[ 4.50062752 15.86818695 9.13286209 
  ...
  4.87327576 2.39396095 18.14508438]
```

Train a regression model with dense_data and times_data

```python
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization

x_train, x_test, y_train, y_test = train_test_split(dense_data, times_data, test_size=0.2, random_state=42)

batch_model = Sequential()
batch_model.add(Dense(2000, input_dim=x_train.shape[1], kernel_initializer='normal', activation='relu'))
batch_model.add(BatchNormalization())
batch_model.add(Dense(2000, kernel_initializer='normal', activation='relu'))
batch_model.add(BatchNormalization())
batch_model.add(Dense(2000, kernel_initializer='normal', activation='relu'))
batch_model.add(BatchNormalization())
batch_model.add(Dense(2000, kernel_initializer='normal', activation='relu'))
batch_model.add(BatchNormalization())
batch_model.add(Dense(2000, kernel_initializer='normal', activation='relu'))
batch_model.add(BatchNormalization())
batch_model.add(Dense(2000, kernel_initializer='normal', activation='relu'))
batch_model.add(BatchNormalization())
batch_model.add(Dense(1, kernel_initializer='normal'))

# Compile model
batch_model.compile(loss='mean_squared_error', optimizer='adam')

history_batch = batch_model.fit(x_train, y_train, batch_size=16, epochs=20, validation_data=(x_test, y_test), verbose=True)
```
```python
# summarize history for loss
plt.plot(history_batch.history['loss'])
plt.plot(history_batch.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train',  'test'], loc='upper left')
plt.show()
```
![enter image description here](https://raw.github.com/aipaca-mlops/ML-training-cost-calculator/create_readme_xin/Images/HistoryDenseUnitSum.png)

```python
batch_y_pred = batch_model.predict(x_test)
batch_y_pred = batch_y_pred.reshape(batch_y_pred.shape[0],  )
plt.scatter(batch_y_pred, y_test)
plt.show()
```
![enter image description here](https://raw.github.com/aipaca-mlops/ML-training-cost-calculator/create_readme_xin/Images/PredDenseUnitSum.png)
</details>  



<details open>  
<summary>Predict CNN Model Training Time Using Model Level Structure</summary>  

Make a prediction for a convolutional network training time.

Build a regression model with generated dataframe data from  **CNN Model Data Generation**.

We first flatten dataframes into training data.

```python  
import numpy as np

x = np.array([
	data_df.to_numpy().reshape(
	model_data_dfs[0].shape[0] * model_data_dfs[0].shape[1],
	)  for data_df in model_data_dfs
])

y = np.array(time_df.batch_time.tolist())
``` 
Build and train regression model. 
```python
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

batch_model = Sequential()
batch_model.add(
Dense(2000, input_dim=x_train.shape[1], kernel_initializer='normal', activation='relu')
)
batch_model.add(BatchNormalization())
batch_model.add(Dense(2000, kernel_initializer='normal', activation='relu'))
batch_model.add(BatchNormalization())
batch_model.add(Dense(2000, kernel_initializer='normal', activation='relu'))
batch_model.add(BatchNormalization())
batch_model.add(Dense(2000, kernel_initializer='normal', activation='relu'))
batch_model.add(BatchNormalization())
batch_model.add(Dense(2000, kernel_initializer='normal', activation='relu'))
batch_model.add(BatchNormalization())
batch_model.add(Dense(2000, kernel_initializer='normal', activation='relu'))
batch_model.add(BatchNormalization())
batch_model.add(Dense(1, kernel_initializer='normal'))
# Compile model
batch_model.compile(loss='mean_squared_error', optimizer='adam')

history_batch = batch_model.fit(
	x_train, y_train, batch_size=16, epochs=20, validation_data=(x_test, y_test), verbose=True
	)
```
</details>  

<details open>  
<summary>Predict CNN Model Training Time Using FLOPs</summary>  

We first get FLOPs features.
```python  
# Convert raw data into data points

# model_data from data generation step for CNN
scaler_conv_data, times_data_conv2d, scaler = ccd.convert_model_config(model_data, layer_num_upper=105, data_type='FLOPs', min_max_scaler=True)
```

Build regression model and train.

```python
# train data

x_train, x_test, y_train, y_test = train_test_split(
scaler_conv_data, times_data_conv2d, test_size=0.1, random_state=0)

batch_model = keras.Sequential()
batch_model.add(Dense(200, kernel_initializer='normal', activation='relu'))
batch_model.add(Dense(200, kernel_initializer='normal', activation='relu'))
batch_model.add(Dense(200, kernel_initializer='normal', activation='relu'))
batch_model.add(Dense(200, kernel_initializer='normal', activation='relu'))
batch_model.add(Dense(200, kernel_initializer='normal', activation='relu'))
batch_model.add(Dense(1, kernel_initializer='normal'))

# Compile model
batch_model.compile(loss='mean_squared_error', optimizer='adam')

 
history_batch = batch_model.fit(
x_train, y_train, batch_size=16, epochs=15, validation_data=(x_test, y_test), verbose=True)

```

</details>  

<details open>  
<summary>Predict RNN Model Training Time</summary>  

Make a prediction for a recurrent network training time.

```python  
put python steps here
```  
</details>  