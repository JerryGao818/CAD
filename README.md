# CAD
Code for WWW'25 paper "Semi-Supervised Anomaly Detection through Denoising-Aware Contrastive Distance Learning".

CAD is implemented using PyTorch. You may need the packages below to run CAD:

- numpy==1.23.1
- torch==1.13.1
- pyod==1.0.0
- scikit-learn==1.0.2
- tqdm==4.65.0

Raw data is accessed from [ODDS dataset](https://odds.cs.stonybrook.edu/) and [ADbench](https://github.com/Minqi824/ADBench?tab=readme-ov-file) and processed using [data_preprocess.py](/data_preprocess.py), and processed data can be found in [processed_data](/processed_data). 

To perfrom anomaly detection, run

    python CAD-main.py --dataset Your-dataset --batch_size batch-size --labeled_ratio ratio-of-labeled-anomalies
