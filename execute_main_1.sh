# First exp (M/N_c)
for idx in 405 #927 3438 405 6438 2169
do
    # # Tau = 0
    # python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000
    # python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 16 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 460000 --test_seq_len 13800000
    # python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 40 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 940000 --test_seq_len 28200000
    # python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 80 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 1740000 --test_seq_len 52200000
    # python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 160 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 3340000 --test_seq_len 100200000
    # python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 400 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 8140000 --test_seq_len 244200000
    # python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 800 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 16140000 --test_seq_len 484200000 

    # python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 64
    # python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 16 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 460000 --test_seq_len 13800000 --bs 64
    # python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 40 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 940000 --test_seq_len 28200000 --bs 64
    # python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 80 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 1740000 --test_seq_len 52200000 --bs 64
    # python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 160 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 3340000 --test_seq_len 100200000 --bs 64
    # python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 400 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 8140000 --test_seq_len 244200000 --bs 64
    # python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 800 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 16140000 --test_seq_len 484200000 --bs 64

    # # Tau = L // 4
    # python main.py --filter_type "LMMSE" --decision_delay 3 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000
    # python main.py --filter_type "LMMSE" --decision_delay 5 --total_taps 8 --filter_size 16 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 460000 --test_seq_len 13800000
    # python main.py --filter_type "LMMSE" --decision_delay 11 --total_taps 8 --filter_size 40 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 940000 --test_seq_len 28200000
    # python main.py --filter_type "LMMSE" --decision_delay 21 --total_taps 8 --filter_size 80 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 1740000 --test_seq_len 52200000
    # python main.py --filter_type "LMMSE" --decision_delay 41 --total_taps 8 --filter_size 160 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 3340000 --test_seq_len 100200000
    # python main.py --filter_type "LMMSE" --decision_delay 101 --total_taps 8 --filter_size 400 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 8140000 --test_seq_len 244200000
    # python main.py --filter_type "LMMSE" --decision_delay 201 --total_taps 8 --filter_size 800 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 16140000 --test_seq_len 484200000 

    # python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 3 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 64
    # python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 5 --total_taps 8 --filter_size 16 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 460000 --test_seq_len 13800000 --bs 64
    # python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 11 --total_taps 8 --filter_size 40 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 940000 --test_seq_len 28200000 --bs 64
    # python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 21 --total_taps 8 --filter_size 80 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 1740000 --test_seq_len 52200000 --bs 64
    # python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 41 --total_taps 8 --filter_size 160 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 3340000 --test_seq_len 100200000 --bs 64
    # python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 101 --total_taps 8 --filter_size 400 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 8140000 --test_seq_len 244200000 --bs 64
    # python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 201 --total_taps 8 --filter_size 800 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 16140000 --test_seq_len 484200000 --bs 64

    #Tau = L // 2
    # python main.py --filter_type "LMMSE" --decision_delay 7 --RX_num 2 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 1
    # python main.py --filter_type "LMMSE" --decision_delay 11 --RX_num 2 --total_taps 8 --filter_size 16 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 460000 --test_seq_len 13800000 --exp_num 1
    # python main.py --filter_type "LMMSE" --decision_delay 23 --RX_num 2 --total_taps 8 --filter_size 40 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 940000 --test_seq_len 28200000 --exp_num 1
    # python main.py --filter_type "LMMSE" --decision_delay 43 --RX_num 2 --total_taps 8 --filter_size 80 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 1740000 --test_seq_len 52200000 --exp_num 1
    # python main.py --filter_type "LMMSE" --decision_delay 83 --RX_num 2 --total_taps 8 --filter_size 160 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 3340000 --test_seq_len 100200000 --exp_num 1
    # python main.py --filter_type "LMMSE" --decision_delay 203 --RX_num 2 --total_taps 8 --filter_size 400 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 8140000 --test_seq_len 244200000 --exp_num 1
    # python main.py --filter_type "LMMSE" --decision_delay 403 --RX_num 2 --total_taps 8 --filter_size 800 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 16140000 --test_seq_len 484200000 --exp_num 1

    # python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 7 --RX_num 2 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 200 --exp_num 1
    # python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 11 --RX_num 2 --total_taps 8 --filter_size 16 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 460000 --test_seq_len 13800000 --bs 200 --exp_num 1 
    # python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 23 --RX_num 2 --total_taps 8 --filter_size 40 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 940000 --test_seq_len 28200000 --bs 200 --exp_num 1 
    # python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 43 --RX_num 2 --total_taps 8 --filter_size 80 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 1740000 --test_seq_len 52200000 --bs 200 --exp_num 1 
    #python main.py --epochs 300 --lr 1e-7 --filter_type "NN" --decision_delay 83 --RX_num 2 --total_taps 8 --filter_size 160 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 3340000 --test_seq_len 100200000 --bs 200 --exp_num 1 
    # python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 203 --RX_num 2 --total_taps 8 --filter_size 400 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 8140000 --test_seq_len 244200000 --bs 200 --exp_num 1 
    # python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 403 --RX_num 2 --total_taps 8 --filter_size 800 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 16140000 --test_seq_len 484200000 --bs 200 --exp_num 1

    #python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 203 --RX_num 2 --total_taps 8 --filter_size 400 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 24420000 --test_seq_len 244200000 --bs 200 --exp_num 1 # 732600000
    python main.py --epochs 360 --lr 1e-7 --filter_type "NN" --decision_delay 403 --RX_num 2 --total_taps 8 --filter_size 800 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 48420000 --test_seq_len 484200000 --bs 200 --exp_num 1 # 1452600000

done

##################################### Viterbi & 16QAM, defense version #####################################

# 16 QAM
# First exp (M/N_c)
# for idx in 927 3438 405 6438 2169 1423 3416 7234 4315 4901 #
# do
#     # Tau = L // 2
#     # python main.py --filter_type "LMMSE" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --mod_scheme 'QPSK' --exp_num 1 #--scatter_plot
#     # python main.py --filter_type "LMMSE" --decision_delay 11 --total_taps 8 --filter_size 16 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 460000 --test_seq_len 13800000 --mod_scheme 'QPSK' --exp_num 1
#     # python main.py --filter_type "LMMSE" --decision_delay 11 --total_taps 8 --filter_size 16 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 920000 --test_seq_len 27600000 --mod_scheme '16QAM' --exp_num 1
#     # python main.py --filter_type "LMMSE" --decision_delay 23 --total_taps 8 --filter_size 40 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 1880000 --test_seq_len 56400000 --mod_scheme '16QAM' --exp_num 1
#     # python main.py --filter_type "LMMSE" --decision_delay 43 --total_taps 8 --filter_size 80 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 3480000 --test_seq_len 104400000 --mod_scheme '16QAM' --exp_num 1
#     # python main.py --filter_type "LMMSE" --decision_delay 83 --total_taps 8 --filter_size 160 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 6680000 --test_seq_len 200400000 --mod_scheme '16QAM' --exp_num 1
#     # python main.py --filter_type "LMMSE" --decision_delay 203 --total_taps 8 --filter_size 400 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 16280000 --test_seq_len 488400000 --mod_scheme '16QAM' --exp_num 1
#     # python main.py --filter_type "LMMSE" --decision_delay 403 --total_taps 8 --filter_size 800 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 32280000 --test_seq_len 968400000 --mod_scheme '16QAM' --exp_num 1

#     # python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 192 --mod_scheme 'QPSK' --exp_num 1 #--scatter_plot
#     # python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 11 --total_taps 8 --filter_size 16 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 920000 --test_seq_len 27600000 --bs 128 --mod_scheme '16QAM' --exp_num 1
#     # python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 23 --total_taps 8 --filter_size 40 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 1880000 --test_seq_len 56400000 --bs 128 --mod_scheme '16QAM' --exp_num 1
#     # python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 43 --total_taps 8 --filter_size 80 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 3480000 --test_seq_len 104400000 --bs 128 --mod_scheme '16QAM' --exp_num 1
#     # python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 83 --total_taps 8 --filter_size 160 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 6680000 --test_seq_len 200400000 --bs 128 --mod_scheme '16QAM' --exp_num 1
#     # python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 203 --total_taps 8 --filter_size 400 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 16280000 --test_seq_len 488400000 --bs 128 --mod_scheme '16QAM' --exp_num 1
#     # python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 403 --total_taps 8 --filter_size 800 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 32280000 --test_seq_len 968400000 --bs 128 --mod_scheme '16QAM' --exp_num 1
# done


# for idx in 927 2169 #3438 405 6438 
# do
#     # Tau = L // 2
#     python main.py --filter_type "LMMSE" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 600000 --test_seq_len 18000000 --mod_scheme '16QAM' --exp_num 1 #--scatter_plot
#     #python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 40 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 600000 --test_seq_len 18000000 --mod_scheme '16QAM' --exp_num 1 --scatter_plot
#     # python main.py --filter_type "LMMSE" --decision_delay 11 --total_taps 8 --filter_size 16 --SNR 10 --rand_seed_channel $idx --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 920000 --test_seq_len 27600000 --mod_scheme '16QAM' --exp_num 1
#     # python main.py --filter_type "LMMSE" --decision_delay 23 --total_taps 8 --filter_size 40 --SNR 10 --rand_seed_channel $idx --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 1880000 --test_seq_len 56400000 --mod_scheme '16QAM' --exp_num 1
#     # python main.py --filter_type "LMMSE" --decision_delay 43 --total_taps 8 --filter_size 80 --SNR 10 --rand_seed_channel $idx --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 3480000 --test_seq_len 104400000 --mod_scheme '16QAM' --exp_num 1
#     # python main.py --filter_type "LMMSE" --decision_delay 83 --total_taps 8 --filter_size 160 --SNR 10 --rand_seed_channel $idx --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 6680000 --test_seq_len 200400000 --mod_scheme '16QAM' --exp_num 1
#     # python main.py --filter_type "LMMSE" --decision_delay 203 --total_taps 8 --filter_size 400 --SNR 10 --rand_seed_channel $idx --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 16280000 --test_seq_len 488400000 --mod_scheme '16QAM' --exp_num 1
#     # python main.py --filter_type "LMMSE" --decision_delay 403 --total_taps 8 --filter_size 800 --SNR 10 --rand_seed_channel $idx --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 32280000 --test_seq_len 968400000 --mod_scheme '16QAM' --exp_num 1

#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 600000 --test_seq_len 18000000 --bs 128 --mod_scheme '16QAM' --exp_num 1 #--scatter_plot
#     # python main.py --epochs 420 --lr 1e-8 --filter_type "NN" --decision_delay 11 --total_taps 8 --filter_size 16 --SNR 10 --rand_seed_channel $idx --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 920000 --test_seq_len 27600000 --bs 192 --mod_scheme '16QAM' --exp_num 1
#     # python main.py --epochs 420 --lr 1e-8 --filter_type "NN" --decision_delay 23 --total_taps 8 --filter_size 40 --SNR 10 --rand_seed_channel $idx --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 1880000 --test_seq_len 56400000 --bs 192 --mod_scheme '16QAM' --exp_num 1
#     # python main.py --epochs 420 --lr 1e-8 --filter_type "NN" --decision_delay 43 --total_taps 8 --filter_size 80 --SNR 10 --rand_seed_channel $idx --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 3480000 --test_seq_len 104400000 --bs 192 --mod_scheme '16QAM' --exp_num 1
#     # python main.py --epochs 420 --lr 1e-8 --filter_type "NN" --decision_delay 83 --total_taps 8 --filter_size 160 --SNR 10 --rand_seed_channel $idx --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 6680000 --test_seq_len 200400000 --bs 192 --mod_scheme '16QAM' --exp_num 1
#     # python main.py --epochs 420 --lr 1e-8 --filter_type "NN" --decision_delay 203 --total_taps 8 --filter_size 400 --SNR 10 --rand_seed_channel $idx --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 16280000 --test_seq_len 488400000 --bs 192 --mod_scheme '16QAM' --exp_num 1
#     # python main.py --epochs 420 --lr 1e-8 --filter_type "NN" --decision_delay 403 --total_taps 8 --filter_size 800 --SNR 10 --rand_seed_channel $idx --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 32280000 --test_seq_len 968400000 --bs 192 --mod_scheme '16QAM' --exp_num 1
# done