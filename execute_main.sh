# Start simulation

python main.py --epochs 5 --lr 1e-3 --filter_type "NN" --filter_size 16 --SNR 20 --rand_seed_train 9999 --rand_seed_test 4999 --train_seq_len 2500 --test_seq_len 250000 --bs 4

python main.py --filter_type "LMMSE" --filter_size 16 --SNR 20 --rand_seed_train 9999 --rand_seed_test 4999 --train_seq_len 2500 --test_seq_len 250000