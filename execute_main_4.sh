# # Fourth exp (SNR)
# for SNR in 0 5 10 15 20 25 30 35 40 # -10 -5
# do
#     for idx in 927	7758	3722	9017	9570	7908	8422	8327	6043	7757	4604	8086	3697	5292	3790	4122	8850	8013	5554	9675
#     do
#         python main.py --filter_type "LMMSE" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 150000 --test_seq_len 9000000 --exp_num 4 # --scatter_plot
#         python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 4
#         python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 200 --exp_num 4 # --scatter_plot
#     done
# done

# # Fourth exp (SNR)
# for SNR in -10 -5 0 5 10 15 20 25 30 35 40
# do
#     for idx in 927 3438 405 6438 2169
#     do
#         python main.py --filter_type "LMMSE" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 150000 --test_seq_len 9000000 --exp_num 4 # --scatter_plot
#         python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 4
#         python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 128 --exp_num 4 # --scatter_plot
#         #python main.py --filter_type "Viterbi" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 6000 --test_seq_len 6000 --exp_num 4
#     done
# done

# Fourth exp (SNR)
for SNR in 0 5 10 15 20 25 30 35 40
do
    for idx in 927 3438 405 6438 2169
    do
        python main.py --filter_type "LMMSE" --decision_delay 11 --RX_num 1 --total_taps 8 --filter_size 16 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 230000 --test_seq_len 13800000 --exp_num 4
        python main.py --filter_type "LS" --decision_delay 11 --RX_num 1 --total_taps 8 --filter_size 16 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 460000 --test_seq_len 13800000 --exp_num 4
        python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 11 --RX_num 1 --total_taps 8 --filter_size 16 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 460000 --test_seq_len 13800000 --bs 200 --exp_num 4
    done
done