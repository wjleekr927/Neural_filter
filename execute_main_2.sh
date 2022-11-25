# Second exp (Training data)

for idx in 927 3438 405 6438 2169
do
    #python main.py --filter_type "LMMSE" --decision_delay 7 --RX_num 2 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 150000 --test_seq_len 9000000 --exp_num 2 # --scatter_plot

    # python main.py --filter_type "LS" --decision_delay 7 --RX_num 2 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 30 --test_seq_len 9000000 --exp_num 2
    # python main.py --filter_type "LS" --decision_delay 7 --RX_num 2 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 60 --test_seq_len 9000000 --exp_num 2
    # python main.py --filter_type "LS" --decision_delay 7 --RX_num 2 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300 --test_seq_len 9000000 --exp_num 2 
    # python main.py --filter_type "LS" --decision_delay 7 --RX_num 2 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 600 --test_seq_len 9000000 --exp_num 2 
    # python main.py --filter_type "LS" --decision_delay 7 --RX_num 2 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 3000 --test_seq_len 9000000 --exp_num 2 
    # python main.py --filter_type "LS" --decision_delay 7 --RX_num 2 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 6000 --test_seq_len 9000000 --exp_num 2
    # python main.py --filter_type "LS" --decision_delay 7 --RX_num 2 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 30000 --test_seq_len 9000000 --exp_num 2 
    # python main.py --filter_type "LS" --decision_delay 7 --RX_num 2 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 60000 --test_seq_len 9000000 --exp_num 2 
    # python main.py --filter_type "LS" --decision_delay 7 --RX_num 2 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 150000 --test_seq_len 9000000 --exp_num 2  # --scatter_plot
    # python main.py --filter_type "LS" --decision_delay 7 --RX_num 2 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 2 
    python main.py --filter_type "LS" --decision_delay 7 --RX_num 2 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 600000 --test_seq_len 9000000 --exp_num 2 
    python main.py --filter_type "LS" --decision_delay 7 --RX_num 2 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 1500000 --test_seq_len 9000000 --exp_num 2 
    python main.py --filter_type "LS" --decision_delay 7 --RX_num 2 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 3000000 --test_seq_len 9000000 --exp_num 2 

    # python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 7 --RX_num 2 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 30 --test_seq_len 9000000 --bs 1 --exp_num 2
    # python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 7 --RX_num 2 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 60 --test_seq_len 9000000 --bs 2 --exp_num 2 
    # python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 7 --RX_num 2 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300 --test_seq_len 9000000 --bs 4 --exp_num 2 
    # python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 7 --RX_num 2 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 600 --test_seq_len 9000000 --bs 8 --exp_num 2 
    # python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 7 --RX_num 2 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 3000 --test_seq_len 9000000 --bs 8 --exp_num 2 
    # python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 7 --RX_num 2 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 6000 --test_seq_len 9000000 --bs 50 --exp_num 2 
    # python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 7 --RX_num 2 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 30000 --test_seq_len 9000000 --bs 100 --exp_num 2 
    # python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 7 --RX_num 2 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 60000 --test_seq_len 9000000 --bs 100 --exp_num 2 
    # python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 7 --RX_num 2 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 150000 --test_seq_len 9000000 --bs 200 --exp_num 2  # --scatter_plot
    # python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 7 --RX_num 2 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 200 --exp_num 2  # --scatter_plot
    python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 7 --RX_num 2 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 600000 --test_seq_len 9000000 --bs 200 --exp_num 2 
    python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 7 --RX_num 2 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 1500000 --test_seq_len 9000000 --bs 200 --exp_num 2 
    python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 7 --RX_num 2 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 3000000 --test_seq_len 9000000 --bs 200 --exp_num 2 

done

##################################### Viterbi & 16QAM, defense version #####################################

# 16 QAM
# Second exp (Training data)
# for idx in 927 3438 405 6438 2169
# do
#     python main.py --filter_type "LMMSE" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 18000000 --mod_scheme '16QAM' --exp_num 2 # --scatter_plot

#     python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 60 --test_seq_len 18000000 --mod_scheme '16QAM' --exp_num 2
#     python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 120 --test_seq_len 18000000 --mod_scheme '16QAM' --exp_num 2
#     python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 600 --test_seq_len 18000000 --mod_scheme '16QAM' --exp_num 2
#     python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 1200 --test_seq_len 18000000 --mod_scheme '16QAM' --exp_num 2
#     python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 6000 --test_seq_len 18000000 --mod_scheme '16QAM' --exp_num 2
#     python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 12000 --test_seq_len 18000000 --mod_scheme '16QAM' --exp_num 2
#     python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 60000 --test_seq_len 18000000 --mod_scheme '16QAM' --exp_num 2
#     python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 120000 --test_seq_len 18000000 --mod_scheme '16QAM' --exp_num 2
#     python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 18000000 --mod_scheme '16QAM' --exp_num 2 # --scatter_plot
#     python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 600000 --test_seq_len 18000000 --mod_scheme '16QAM' --exp_num 2

#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 60 --test_seq_len 18000000 --bs 1 --mod_scheme '16QAM' --exp_num 2
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 120 --test_seq_len 18000000 --bs 2 --mod_scheme '16QAM' --exp_num 2
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 600 --test_seq_len 18000000 --bs 4 --mod_scheme '16QAM' --exp_num 2
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 1200 --test_seq_len 18000000 --bs 8 --mod_scheme '16QAM' --exp_num 2
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 6000 --test_seq_len 18000000 --bs 8 --mod_scheme '16QAM' --exp_num 2
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 12000 --test_seq_len 18000000 --bs 16 --mod_scheme '16QAM' --exp_num 2
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 60000 --test_seq_len 18000000 --bs 16 --mod_scheme '16QAM' --exp_num 2
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 120000 --test_seq_len 18000000 --bs 32 --mod_scheme '16QAM' --exp_num 2
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 18000000 --bs 32 --mod_scheme '16QAM' --exp_num 2 # --scatter_plot
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 600000 --test_seq_len 18000000 --bs 128 --mod_scheme '16QAM' --exp_num 2 # --scatter_plot
# done