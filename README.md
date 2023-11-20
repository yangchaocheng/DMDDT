# DDTM: Denoising Diffusion Transformer Models for Multivariate Time Series Anomaly Detection



**Abstract:** Anomaly detection in multivariate time series has emerged as a crucial challenge in time
series research, with significant research implications in various fields such as fraud detection,
fault diagnosis, and system state estimation. Reconstruction-based models have shown promising potential in recent years for detecting anomalies in time series data. However, due to the
rapid increase in data scale and dimensionality, the issues of noise and identical shortcut during
time series reconstruction have become increasingly pronounced. In this context, we propose a
novel framework for multivariate Time Series Anomaly Detection(TSAD) by introducing the
novel Neighbor Mask Transformer (NMA) model and integrating it with the Denoising Diffusion Model (DDM). The NMA incorporates a neighborhood mask attention module to prevent
information leakage during data reconstruction, effectively addressing the identical shortcut
problem. The Denoising Diffusion Model employs a step-by-step generation process to model
the probability distribution of time series data, to capture patterns in normal data and progressively restoring time series by eliminating noise, thereby facilitating precise identification
and recognition of anomalous points. Our proposed model has been thoroughly evaluated
on five publicly available multivariate TSAD datasets. The experimental results demonstrate
that the model can effectively identify anomalies in time series data, achieving state-of-the-art
performance in anomaly detection.

## Requirements

* Python 3.9
* PyTorch version 1.13.1+cu117
* numpy
* scipy
* pandas
* Pillow
* scikit-learn
* xlrd
## Dependencies can be installed using the following command:

```
pip install -r requirements.txt
```

## Datasets

MSL, PSM, SMAP, SMD, and SWaT datasets were acquired at: [datasets](https://drive.google.com/drive/folders/1q_oXl7xoyNQdcNhPkP9aRXnrkGrFPhHu?usp=sharing). 

- Create the Folder 'datasets'.
- unzip the 'datasets' on the datasets folder.

Then you can get the folder tree shown as below:

```
|DDTM
| |-datasets
| | |-MSL
| | | |-MSL_test.npy
| | | |-MSL_test_label.npy
| | | |-MSL_train.npy
| | |
| | |-PSM
| | | |-test.csv
| | | |-test_label.csv
| | | |-train
| | |
| | |-SMAP
| | | |-SMAP_test.npy
| | | |-SMAP_test_label.npy
| | | |-SMAP_train.npy
| | |
| | |-SMD
| | | |-SMD_test.npy
| | | |-SMD_test.pkl
| | | |-SMD_test_label.npy
| | | |-SMD_test_label.pkl
| | | |-SMD_train.npy
| | | |-SMD_train.pkl
| | |
| | |-SWaT
| | | |-SWaT_Dataset_Attack_v0.xlsx
| | | |-SWaT_Dataset_Normal_v1.xlsx
| | | |-swat_raw.csv
| | | |-swat_train.csv
| | | |-swat_train2.csv
| | | |-swat2.csv
```

## Usage
Commands for training and testing FDNet of all datasets are in `./scripts/DDTM.sh`.

More parameter information please refer to `main.py`.

We provide a complete command for training and testing DDTM:

```
python main.py --anomaly_ratio <anomaly_ratio> --num_epochs <num_epochs>   --batch_size <batch_size>  --mode <mode>  --dataset <dataset>   --data_path <data_path>     --input_c <input_c>   --output_c <output_c>
```

## Main Result

Table presents a comprehensive performance comparison between our model and other baseline models. The best results are indicated in bold black, while the second-best results are represented with underlines.

<p align="center">
<img src="./pics/result.png" height = "300" alt="" align=center />
<br><br>
<b>Figure 1.</b> Results
</p>