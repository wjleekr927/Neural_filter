##################################### Viterbi & 16QAM, defense version #####################################

# 16 QAM
# First exp (M/N_c)
for idx in 927 3438 405 6438 2169
do
    # Tau = L // 2
    python main.py --filter_type "LMMSE" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 600000 --test_seq_len 18000000 --mod_scheme '16QAM' --exp_num 1
    python main.py --filter_type "LMMSE" --decision_delay 11 --total_taps 8 --filter_size 16 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 920000 --test_seq_len 27600000 --mod_scheme '16QAM' --exp_num 1
    python main.py --filter_type "LMMSE" --decision_delay 23 --total_taps 8 --filter_size 40 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 1880000 --test_seq_len 56400000 --mod_scheme '16QAM' --exp_num 1
    python main.py --filter_type "LMMSE" --decision_delay 43 --total_taps 8 --filter_size 80 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 3480000 --test_seq_len 104400000 --mod_scheme '16QAM' --exp_num 1
    python main.py --filter_type "LMMSE" --decision_delay 83 --total_taps 8 --filter_size 160 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 6680000 --test_seq_len 200400000 --mod_scheme '16QAM' --exp_num 1
    python main.py --filter_type "LMMSE" --decision_delay 203 --total_taps 8 --filter_size 400 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 16280000 --test_seq_len 488400000 --mod_scheme '16QAM' --exp_num 1
    python main.py --filter_type "LMMSE" --decision_delay 403 --total_taps 8 --filter_size 800 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 32280000 --test_seq_len 968400000 --mod_scheme '16QAM' --exp_num 1

    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 600000 --test_seq_len 18000000 --bs 128 --mod_scheme '16QAM' --exp_num 1
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 11 --total_taps 8 --filter_size 16 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 920000 --test_seq_len 27600000 --bs 128 --mod_scheme '16QAM' --exp_num 1
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 23 --total_taps 8 --filter_size 40 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 1880000 --test_seq_len 56400000 --bs 128 --mod_scheme '16QAM' --exp_num 1
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 43 --total_taps 8 --filter_size 80 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 3480000 --test_seq_len 104400000 --bs 128 --mod_scheme '16QAM' --exp_num 1
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 83 --total_taps 8 --filter_size 160 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 6680000 --test_seq_len 200400000 --bs 128 --mod_scheme '16QAM' --exp_num 1
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 203 --total_taps 8 --filter_size 400 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 16280000 --test_seq_len 488400000 --bs 128 --mod_scheme '16QAM' --exp_num 1
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 403 --total_taps 8 --filter_size 800 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 32280000 --test_seq_len 968400000 --bs 128 --mod_scheme '16QAM' --exp_num 1
done