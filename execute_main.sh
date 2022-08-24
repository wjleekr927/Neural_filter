# Start simulation
# Control SNR and decision_delay by bash variable

# First variable can be used as $1.
# e.g.) SNR=$1

######################### 8 taps #########################
# Drop 2, 9 th seed
# 2169 5551 6438 0405 3438 2077 0927 7522 9716 8837
# 2169 6438 0405 3438 2077 0927 7522 8837 (for 8 averaging)

# First exp (M/N_c)

for idx in 2169 5551 6438 0405 3438 2077 0927 7522 9716 8837
do
#     # Tau = 0
#     python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000
#     python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 16 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 460000 --test_seq_len 13800000
#     python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 40 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 940000 --test_seq_len 28200000
#     python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 80 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 1740000 --test_seq_len 52200000
#     python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 160 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 3340000 --test_seq_len 100200000
#     python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 400 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 8140000 --test_seq_len 244200000
#     python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 800 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 16140000 --test_seq_len 484200000 

#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 64
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 16 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 460000 --test_seq_len 13800000 --bs 64
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 40 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 940000 --test_seq_len 28200000 --bs 64
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 80 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 1740000 --test_seq_len 52200000 --bs 64
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 160 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 3340000 --test_seq_len 100200000 --bs 64
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 400 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 8140000 --test_seq_len 244200000 --bs 64
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 800 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 16140000 --test_seq_len 484200000 --bs 64

#     # Tau = L // 4
#     python main.py --filter_type "LMMSE" --decision_delay 3 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000
#     python main.py --filter_type "LMMSE" --decision_delay 5 --total_taps 8 --filter_size 16 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 460000 --test_seq_len 13800000
#     python main.py --filter_type "LMMSE" --decision_delay 11 --total_taps 8 --filter_size 40 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 940000 --test_seq_len 28200000
#     python main.py --filter_type "LMMSE" --decision_delay 21 --total_taps 8 --filter_size 80 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 1740000 --test_seq_len 52200000
#     python main.py --filter_type "LMMSE" --decision_delay 41 --total_taps 8 --filter_size 160 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 3340000 --test_seq_len 100200000
#     python main.py --filter_type "LMMSE" --decision_delay 101 --total_taps 8 --filter_size 400 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 8140000 --test_seq_len 244200000
#     python main.py --filter_type "LMMSE" --decision_delay 201 --total_taps 8 --filter_size 800 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 16140000 --test_seq_len 484200000 

#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 3 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 64
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 5 --total_taps 8 --filter_size 16 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 460000 --test_seq_len 13800000 --bs 64
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 11 --total_taps 8 --filter_size 40 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 940000 --test_seq_len 28200000 --bs 64
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 21 --total_taps 8 --filter_size 80 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 1740000 --test_seq_len 52200000 --bs 64
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 41 --total_taps 8 --filter_size 160 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 3340000 --test_seq_len 100200000 --bs 64
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 101 --total_taps 8 --filter_size 400 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 8140000 --test_seq_len 244200000 --bs 64
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 201 --total_taps 8 --filter_size 800 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 16140000 --test_seq_len 484200000 --bs 64

    # Tau = L // 2
    python main.py --filter_type "LMMSE" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000
    python main.py --filter_type "LMMSE" --decision_delay 11 --total_taps 8 --filter_size 16 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 460000 --test_seq_len 13800000
    python main.py --filter_type "LMMSE" --decision_delay 23 --total_taps 8 --filter_size 40 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 940000 --test_seq_len 28200000
    python main.py --filter_type "LMMSE" --decision_delay 43 --total_taps 8 --filter_size 80 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 1740000 --test_seq_len 52200000
    python main.py --filter_type "LMMSE" --decision_delay 83 --total_taps 8 --filter_size 160 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 3340000 --test_seq_len 100200000
    python main.py --filter_type "LMMSE" --decision_delay 203 --total_taps 8 --filter_size 400 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 8140000 --test_seq_len 244200000
    python main.py --filter_type "LMMSE" --decision_delay 403 --total_taps 8 --filter_size 800 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 16140000 --test_seq_len 484200000 

    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 64
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 11 --total_taps 8 --filter_size 16 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 460000 --test_seq_len 13800000 --bs 64
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 23 --total_taps 8 --filter_size 40 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 940000 --test_seq_len 28200000 --bs 64
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 43 --total_taps 8 --filter_size 80 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 1740000 --test_seq_len 52200000 --bs 64
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 83 --total_taps 8 --filter_size 160 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 3340000 --test_seq_len 100200000 --bs 64
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 203 --total_taps 8 --filter_size 400 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 8140000 --test_seq_len 244200000 --bs 64
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 403 --total_taps 8 --filter_size 800 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 16140000 --test_seq_len 484200000 --bs 64

done





# Second exp (Training data)

for idx in 0927
do
    python main.py --filter_type "LMMSE" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 150000 --test_seq_len 9000000 # --scatter_plot

    python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 30 --test_seq_len 9000000
    python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 60 --test_seq_len 9000000
    python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300 --test_seq_len 9000000
    python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 600 --test_seq_len 9000000 
    python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 3000 --test_seq_len 9000000
    python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 6000 --test_seq_len 9000000
    python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 30000 --test_seq_len 9000000
    python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 60000 --test_seq_len 9000000
    python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 150000 --test_seq_len 9000000 # --scatter_plot
    python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 

    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 30 --test_seq_len 9000000 --bs 1
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 60 --test_seq_len 9000000 --bs 2
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300 --test_seq_len 9000000 --bs 4
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 600 --test_seq_len 9000000 --bs 8
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 3000 --test_seq_len 9000000 --bs 8
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 6000 --test_seq_len 9000000 --bs 16
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 30000 --test_seq_len 9000000 --bs 16
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 60000 --test_seq_len 9000000 --bs 32
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 150000 --test_seq_len 9000000 --bs 32 # --scatter_plot
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 128 # --scatter_plot
done


# Third exp (SNR)

for SNR in -10 -5 0 5 10 15 20 25 30 35 40
do
    for idx in 0927
    do
        python main.py --filter_type "LMMSE" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 150000 --test_seq_len 9000000 # --scatter_plot

        # python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 30 --test_seq_len 9000000
        # python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 60 --test_seq_len 9000000
        # python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300 --test_seq_len 9000000
        # python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 600 --test_seq_len 9000000 
        # python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 3000 --test_seq_len 9000000
        # python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 6000 --test_seq_len 9000000
        # python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 30000 --test_seq_len 9000000
        # python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 60000 --test_seq_len 9000000
        # python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 150000 --test_seq_len 9000000 # --scatter_plot
        python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 

        # python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 30 --test_seq_len 9000000 --bs 1
        # python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 60 --test_seq_len 9000000 --bs 2
        # python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300 --test_seq_len 9000000 --bs 4
        # python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 600 --test_seq_len 9000000 --bs 8
        # python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 3000 --test_seq_len 9000000 --bs 8
        # python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 6000 --test_seq_len 9000000 --bs 16
        # python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 30000 --test_seq_len 9000000 --bs 16
        # python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 60000 --test_seq_len 9000000 --bs 32
        # python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 150000 --test_seq_len 9000000 --bs 32 --scatter_plot
        python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 128 # --scatter_plot
    done
done




# Fourth exp (tau)

for idx in 7406 5725 7416 7213 3364 2990 3588 4540 7449 5978 4477 7944 2047 3039 3638 5114 8997 6291 1580 5297 5268 5462 2312 5054 3879 6244 3048 8053 8996 3821 8056 3720
do
    # Tau = 0
    python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000
    python main.py --filter_type "LS" --decision_delay 0 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 128

    python main.py --filter_type "LMMSE" --decision_delay 1 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000
    python main.py --filter_type "LS" --decision_delay 1 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 1 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 128

    python main.py --filter_type "LMMSE" --decision_delay 2 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000
    python main.py --filter_type "LS" --decision_delay 2 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 2 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 128

    # Tau = L // 4
    python main.py --filter_type "LMMSE" --decision_delay 3 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000
    python main.py --filter_type "LS" --decision_delay 3 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 3 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 128

    python main.py --filter_type "LMMSE" --decision_delay 4 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000
    python main.py --filter_type "LS" --decision_delay 4 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 4 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 128

    python main.py --filter_type "LMMSE" --decision_delay 5 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000
    python main.py --filter_type "LS" --decision_delay 5 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 5 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 128

    python main.py --filter_type "LMMSE" --decision_delay 6 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000
    python main.py --filter_type "LS" --decision_delay 6 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 6 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 128

    # Tau = L // 2
    python main.py --filter_type "LMMSE" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000
    python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 128

    python main.py --filter_type "LMMSE" --decision_delay 8 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000
    python main.py --filter_type "LS" --decision_delay 8 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 8 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 128

    python main.py --filter_type "LMMSE" --decision_delay 9 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000
    python main.py --filter_type "LS" --decision_delay 9 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 9 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 128
    
    python main.py --filter_type "LMMSE" --decision_delay 10 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000
    python main.py --filter_type "LS" --decision_delay 10 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 10 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 128

    python main.py --filter_type "LMMSE" --decision_delay 11 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000
    python main.py --filter_type "LS" --decision_delay 11 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 11 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 128

    python main.py --filter_type "LMMSE" --decision_delay 12 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000
    python main.py --filter_type "LS" --decision_delay 12 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 12 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 128

    python main.py --filter_type "LMMSE" --decision_delay 13 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000
    python main.py --filter_type "LS" --decision_delay 13 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 13 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 128

    python main.py --filter_type "LMMSE" --decision_delay 14 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000
    python main.py --filter_type "LS" --decision_delay 14 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 14 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 128

done