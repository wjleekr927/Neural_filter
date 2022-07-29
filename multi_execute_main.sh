# 임시로 second exp 돌리는 중입니다.

SNR=$1

# Second exp.
for idx in 2169 5551 6438 0405 3438 2077 0927 7522 9716 8837
do
    # # Tau = 0
    # python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000
    # python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 16 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 460000 --test_seq_len 13800000
    # python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 40 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 940000 --test_seq_len 28200000
    # python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 80 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 1740000 --test_seq_len 52200000
    # python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 160 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 3340000 --test_seq_len 100200000
    # python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 400 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 8140000 --test_seq_len 244200000
    # python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 800 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 16140000 --test_seq_len 484200000 

    # python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 64
    # python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 16 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 460000 --test_seq_len 13800000 --bs 64
    # python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 40 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 940000 --test_seq_len 28200000 --bs 64
    # python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 80 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 1740000 --test_seq_len 52200000 --bs 64
    # python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 160 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 3340000 --test_seq_len 100200000 --bs 64
    # python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 400 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 8140000 --test_seq_len 244200000 --bs 64
    # python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 800 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 16140000 --test_seq_len 484200000 --bs 64

    # Tau = L // 4
    python main.py --filter_type "LMMSE" --decision_delay 3 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000
    python main.py --filter_type "LMMSE" --decision_delay 5 --total_taps 8 --filter_size 16 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 460000 --test_seq_len 13800000
    python main.py --filter_type "LMMSE" --decision_delay 11 --total_taps 8 --filter_size 40 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 940000 --test_seq_len 28200000
    python main.py --filter_type "LMMSE" --decision_delay 21 --total_taps 8 --filter_size 80 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 1740000 --test_seq_len 52200000
    python main.py --filter_type "LMMSE" --decision_delay 41 --total_taps 8 --filter_size 160 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 3340000 --test_seq_len 100200000
    python main.py --filter_type "LMMSE" --decision_delay 101 --total_taps 8 --filter_size 400 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 8140000 --test_seq_len 244200000
    python main.py --filter_type "LMMSE" --decision_delay 201 --total_taps 8 --filter_size 800 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 16140000 --test_seq_len 484200000 

    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 3 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 64
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 5 --total_taps 8 --filter_size 16 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 460000 --test_seq_len 13800000 --bs 64
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 11 --total_taps 8 --filter_size 40 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 940000 --test_seq_len 28200000 --bs 64
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 21 --total_taps 8 --filter_size 80 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 1740000 --test_seq_len 52200000 --bs 64
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 41 --total_taps 8 --filter_size 160 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 3340000 --test_seq_len 100200000 --bs 64
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 101 --total_taps 8 --filter_size 400 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 8140000 --test_seq_len 244200000 --bs 64
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 201 --total_taps 8 --filter_size 800 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 16140000 --test_seq_len 484200000 --bs 64

    # Tau = L // 2
    python main.py --filter_type "LMMSE" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000
    python main.py --filter_type "LMMSE" --decision_delay 11 --total_taps 8 --filter_size 16 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 460000 --test_seq_len 13800000
    python main.py --filter_type "LMMSE" --decision_delay 23 --total_taps 8 --filter_size 40 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 940000 --test_seq_len 28200000
    python main.py --filter_type "LMMSE" --decision_delay 43 --total_taps 8 --filter_size 80 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 1740000 --test_seq_len 52200000
    python main.py --filter_type "LMMSE" --decision_delay 83 --total_taps 8 --filter_size 160 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 3340000 --test_seq_len 100200000
    python main.py --filter_type "LMMSE" --decision_delay 203 --total_taps 8 --filter_size 400 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 8140000 --test_seq_len 244200000
    python main.py --filter_type "LMMSE" --decision_delay 403 --total_taps 8 --filter_size 800 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 16140000 --test_seq_len 484200000 

    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 64
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 11 --total_taps 8 --filter_size 16 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 460000 --test_seq_len 13800000 --bs 64
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 23 --total_taps 8 --filter_size 40 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 940000 --test_seq_len 28200000 --bs 64
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 43 --total_taps 8 --filter_size 80 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 1740000 --test_seq_len 52200000 --bs 64
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 83 --total_taps 8 --filter_size 160 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 3340000 --test_seq_len 100200000 --bs 64
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 203 --total_taps 8 --filter_size 400 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 8140000 --test_seq_len 244200000 --bs 64
    python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 403 --total_taps 8 --filter_size 800 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 16140000 --test_seq_len 484200000 --bs 64

done