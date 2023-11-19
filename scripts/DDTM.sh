# SMD
python main.py --anomaly_ratio 0.5 --num_epochs 100   --batch_size 256  --mode train   --dataset SMD   --data_path datasets/SMD     --input_c 38   --output_c 38
python main.py --anomaly_ratio 0.5 --num_epochs 100   --batch_size 256  --mode test    --dataset SMD   --data_path datasets/SMD     --input_c 38   --output_c 38
# SWaT
python main.py --anomaly_ratio 0.5  --num_epochs 100    --batch_size 256  --mode train   --dataset SWaT   --data_path datasets/SWaT   --input_c 51    --output_c 51
python main.py --anomaly_ratio 0.5  --num_epochs 100    --batch_size 256  --mode test    --dataset SWaT   --data_path datasets/SWaT   --input_c 51    --output_c 51
# MSL
python main.py --anomaly_ratio 1.0  --num_epochs 100   --batch_size 256  --mode train   --dataset MSL   --data_path datasets/MSL   --input_c 55    --output_c 55
python main.py --anomaly_ratio 1.0  --num_epochs 100   --batch_size 256  --mode test    --dataset MSL   --data_path datasets/MSL   --input_c 55    --output_c 55
# SMAP
python main.py --anomaly_ratio 1.0  --num_epochs 100   --batch_size 256  --mode train   --dataset SMAP   --data_path datasets/SMAP   --input_c 25    --output_c 25
python main.py --anomaly_ratio 1.0  --num_epochs 100   --batch_size 256  --mode test    --dataset SMAP   --data_path datasets/SMAP   --input_c 25    --output_c 25
# PSM
python main.py --anomaly_ratio 1.0  --num_epochs 100    --batch_size 256  --mode train   --dataset PSM   --data_path datasets/PSM   --input_c 25    --output_c 25
python main.py --anomaly_ratio 1.0  --num_epochs 100    --batch_size 256  --mode test    --dataset PSM   --data_path datasets/PSM   --input_c 25    --output_c 25


