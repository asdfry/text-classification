rm -r logs
mkdir logs
accelerate launch --config_file ./configs/gpu_4.yaml train_multi.py -e 5 -b 8 -g 4 -t -hf
accelerate launch --config_file ./configs/gpu_2.yaml train_multi.py -e 5 -b 8 -g 2 -t -hf
accelerate launch --config_file ./configs/gpu_1.yaml train_multi.py -e 5 -b 8 -g 1 -t -hf
python parse_logs.py
