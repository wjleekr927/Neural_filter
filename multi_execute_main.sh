# for idx in 927	7758	3722	9017	#9570	7908	8422	8327	6043	7757
# do
#     python main.py --filter_type "LMMSE" --decision_delay 11 --total_taps 12 --filter_size 12 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 920000 --test_seq_len 27600000 --mod_scheme '16QAM' --exp_num 0
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 11 --total_taps 12 --filter_size 12 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 920000 --test_seq_len 27600000 --bs 256 --mod_scheme '16QAM' --exp_num 0 
#     #python main.py --filter_type "Viterbi" --decision_delay 11 --total_taps 12 --filter_size 12 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 2300 --test_seq_len 2300 --mod_scheme 'QPSK' --exp_num 0
# done

# for idx in 927	7758	3722	9017	9570	7908	8422	8327	6043	7757
# do
#     # python main.py --filter_type "LMMSE" --decision_delay 11 --total_taps 12 --filter_size 12 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 920000 --test_seq_len 27600000 --mod_scheme '16QAM' --exp_num 0
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 11 --total_taps 12 --filter_size 12 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 920000 --test_seq_len 27600000 --bs 200 --mod_scheme '16QAM' --exp_num 0 
#     # python main.py --filter_type "Viterbi" --decision_delay 11 --total_taps 12 --filter_size 12 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 2300 --test_seq_len 2300 --mod_scheme '16QAM' --exp_num 0
# done

# Scatter plot (for defense)
for idx in 927
do
    # python main.py --filter_type "LMMSE" --decision_delay 7 --RX_num 2 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 1 --scatter_plot
    # python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 7 --RX_num 2 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 200 --exp_num 1 --scatter_plot
    #python main.py --filter_type "LMMSE" --decision_delay 7 --RX_num 2 --mod_scheme '16QAM' --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 600000 --test_seq_len 18000000 --exp_num 1 --scatter_plot
    python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 7 --mod_scheme '16QAM' --RX_num 2 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 1800000 --test_seq_len 18000000 --bs 200 --exp_num 1 --scatter_plot
done