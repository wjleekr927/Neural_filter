# Start simulation

# python main.py --epochs 10 --lr 1e-3 --filter_type "NN" --filter_size 16 --SNR -30 --rand_seed_train 9925 --rand_seed_test 4925 --train_seq_len 2500 --test_seq_len 250000 --bs 4
# python main.py --epochs 10 --lr 1e-3 --filter_type "NN" --filter_size 16 --SNR -15 --rand_seed_train 9925 --rand_seed_test 4925 --train_seq_len 2500 --test_seq_len 250000 --bs 4
# python main.py --epochs 10 --lr 1e-3 --filter_type "NN" --filter_size 16 --SNR 0 --rand_seed_train 9925 --rand_seed_test 4925 --train_seq_len 2500 --test_seq_len 250000 --bs 4
# python main.py --epochs 10 --lr 1e-3 --filter_type "NN" --filter_size 16 --SNR 15 --rand_seed_train 9925 --rand_seed_test 4925 --train_seq_len 2500 --test_seq_len 250000 --bs 4
# python main.py --epochs 10 --lr 1e-3 --filter_type "NN" --filter_size 16 --SNR 30 --rand_seed_train 9925 --rand_seed_test 4925 --train_seq_len 2500 --test_seq_len 250000 --bs 4

# python main.py --filter_type "LMMSE" --decay_factor 0 --filter_size 16 --SNR -30 --rand_seed_train 9925 --rand_seed_test 4925 --train_seq_len 2500 --test_seq_len 250000
# python main.py --filter_type "LMMSE" --decay_factor 0 --filter_size 16 --SNR -15 --rand_seed_train 9925 --rand_seed_test 4925 --train_seq_len 2500 --test_seq_len 250000
# python main.py --filter_type "LMMSE" --decay_factor 0 --filter_size 16 --SNR 0 --rand_seed_train 9925 --rand_seed_test 4925 --train_seq_len 2500 --test_seq_len 250000
# python main.py --filter_type "LMMSE" --decay_factor 0 --filter_size 16 --SNR 15 --rand_seed_train 9925 --rand_seed_test 4925 --train_seq_len 2500 --test_seq_len 250000
# python main.py --filter_type "LMMSE" --decay_factor 0 --filter_size 16 --SNR 30 --rand_seed_train 9925 --rand_seed_test 4925 --train_seq_len 2500 --test_seq_len 250000

# python main.py --filter_type "LMMSE" --decay_factor 0 --filter_size 90 --SNR -30 --rand_seed_train 9999 --rand_seed_test 4999 --train_seq_len 9900 --test_seq_len 990000

# python main.py --epochs 10 --lr 1e-4 --filter_type "NN" --total_taps 10 --decay_factor 0 --filter_size 10 --SNR 20 --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 19000 --test_seq_len 190000 --bs 4
python main.py --epochs 10 --lr 1e-4 --filter_type "NN" --total_taps 10 --decay_factor 0 --filter_size 20 --SNR 20 --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 29000 --test_seq_len 290000 --bs 4
python main.py --epochs 10 --lr 1e-4 --filter_type "NN" --total_taps 10 --decay_factor 0 --filter_size 50 --SNR 20 --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 59000 --test_seq_len 590000 --bs 4
python main.py --epochs 10 --lr 1e-4 --filter_type "NN" --total_taps 10 --decay_factor 0 --filter_size 100 --SNR 20 --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 109000 --test_seq_len 1090000 --bs 4
python main.py --epochs 10 --lr 1e-4 --filter_type "NN" --total_taps 10 --decay_factor 0 --filter_size 200 --SNR 20 --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 209000 --test_seq_len 2090000 --bs 4
python main.py --epochs 10 --lr 1e-4 --filter_type "NN" --total_taps 10 --decay_factor 0 --filter_size 500 --SNR 20 --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 509000 --test_seq_len 5090000 --bs 4
python main.py --epochs 10 --lr 1e-4 --filter_type "NN" --total_taps 10 --decay_factor 0 --filter_size 1000 --SNR 20 --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 1009000 --test_seq_len 10090000 --bs 4

# python main.py --filter_type "LMMSE" --total_taps 10 --decay_factor 0 --filter_size 10 --SNR 40 --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 19000 --test_seq_len 190000
# python main.py --filter_type "LMMSE" --total_taps 10 --decay_factor 0 --filter_size 20 --SNR 40 --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 29000 --test_seq_len 290000
# python main.py --filter_type "LMMSE" --total_taps 10 --decay_factor 0 --filter_size 50 --SNR 40 --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 59000 --test_seq_len 590000
# python main.py --filter_type "LMMSE" --total_taps 10 --decay_factor 0 --filter_size 100 --SNR 40 --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 109000 --test_seq_len 1090000
# python main.py --filter_type "LMMSE" --total_taps 10 --decay_factor 0 --filter_size 200 --SNR 40 --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 209000 --test_seq_len 2090000
# python main.py --filter_type "LMMSE" --total_taps 10 --decay_factor 0 --filter_size 500 --SNR 40 --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 509000 --test_seq_len 5090000
# python main.py --filter_type "LMMSE" --total_taps 10 --decay_factor 0 --filter_size 1000 --SNR 40 --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 1009000 --test_seq_len 10090000