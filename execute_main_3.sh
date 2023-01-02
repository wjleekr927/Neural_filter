# Third exp (tau)

# for idx in 927	7758	3722	9017	9570	7908	8422	8327	6043	7757	4604	8086	3697	5292	3790	4122	8850	8013	5554	9675
# do
#     # Tau = 0
#     python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 3
#     python main.py --filter_type "LS" --decision_delay 0 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 3
#     python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 200 --exp_num 3

#     python main.py --filter_type "LMMSE" --decision_delay 1 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 3
#     python main.py --filter_type "LS" --decision_delay 1 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 3
#     python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 1 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 200 --exp_num 3

#     python main.py --filter_type "LMMSE" --decision_delay 2 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 3
#     python main.py --filter_type "LS" --decision_delay 2 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 3
#     python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 2 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 200 --exp_num 3

#     # Tau = L // 4
#     python main.py --filter_type "LMMSE" --decision_delay 3 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 3
#     python main.py --filter_type "LS" --decision_delay 3 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 3
#     python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 3 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 200 --exp_num 3

#     python main.py --filter_type "LMMSE" --decision_delay 4 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 3
#     python main.py --filter_type "LS" --decision_delay 4 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 3
#     python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 4 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 200 --exp_num 3

#     python main.py --filter_type "LMMSE" --decision_delay 5 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 3
#     python main.py --filter_type "LS" --decision_delay 5 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 3
#     python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 5 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 200 --exp_num 3

#     python main.py --filter_type "LMMSE" --decision_delay 6 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 3
#     python main.py --filter_type "LS" --decision_delay 6 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 3
#     python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 6 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 200 --exp_num 3

#     # Tau = L // 2
#     python main.py --filter_type "LMMSE" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 3
#     python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 3
#     python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 200 --exp_num 3

#     python main.py --filter_type "LMMSE" --decision_delay 8 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 3
#     python main.py --filter_type "LS" --decision_delay 8 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 3
#     python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 8 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 200 --exp_num 3

#     python main.py --filter_type "LMMSE" --decision_delay 9 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 3
#     python main.py --filter_type "LS" --decision_delay 9 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 3
#     python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 9 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 200 --exp_num 3
    
#     python main.py --filter_type "LMMSE" --decision_delay 10 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 3
#     python main.py --filter_type "LS" --decision_delay 10 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 3
#     python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 10 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 200 --exp_num 3

#     python main.py --filter_type "LMMSE" --decision_delay 11 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 3
#     python main.py --filter_type "LS" --decision_delay 11 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 3
#     python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 11 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 200 --exp_num 3

#     python main.py --filter_type "LMMSE" --decision_delay 12 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 3
#     python main.py --filter_type "LS" --decision_delay 12 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 3
#     python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 12 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 200 --exp_num 3

#     python main.py --filter_type "LMMSE" --decision_delay 13 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 3
#     python main.py --filter_type "LS" --decision_delay 13 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 3
#     python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 13 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 200 --exp_num 3

#     python main.py --filter_type "LMMSE" --decision_delay 14 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 3
#     python main.py --filter_type "LS" --decision_delay 14 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 3
#     python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 14 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 200 --exp_num 3

# done

# Third exp (tau)
for tau in 6 #0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
do
    for idx in 927 #927 3438 405 6438 2169
    do
        #python main.py --filter_type "Viterbi" --decision_delay $tau --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 6000 --test_seq_len 6000 --exp_num 3
        #python main.py --filter_type "LMMSE" --RX_num 2 --decision_delay $tau --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 3
        #python main.py --filter_type "LS" --RX_num 2 --decision_delay $tau --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 3
        python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --RX_num 2 --decision_delay $tau --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 200 --exp_num 3
    done
done