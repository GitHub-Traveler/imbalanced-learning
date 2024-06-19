#!/bin/bash
config = "resnet18_balance"


python -m scripts.02_stats_to_data.training_run_json_to_csv --input_dir ../test_0.01/json/resnet18_0.1imb/ --save_dir ../test/csv/resnet18_0.1imb --has_loss --exp_type cnn

python -m scripts.04_train_hmm.model_selection --data_dir ../results_1e-3/csv/resnet18_0.1imb/ --output_file ../results_1e-3/hmm/resnet18_0.1imb --dataset_name cifar100 --exp_type base --cov_type full --num_iters 32 --max_components 8