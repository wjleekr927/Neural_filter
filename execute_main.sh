# Start simulation

######################### 2 taps #########################
# python main.py --epochs 10 --lr 1e-4 --filter_type "NN" --decision_delay 1 --total_taps 2 --filter_size 2 --SNR 20 --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 3000 --test_seq_len 30000 --bs 2
# python main.py --epochs 10 --lr 1e-4 --filter_type "NN" --decision_delay 2 --total_taps 2 --filter_size 2 --SNR 20 --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 3000 --test_seq_len 30000 --bs 2

# python main.py --epochs 10 --lr 1e-4 --filter_type "NN" --decision_delay 1 --total_taps 2 --filter_size 4 --SNR 20 --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 5000 --test_seq_len 50000 --bs 2
# python main.py --epochs 10 --lr 1e-4 --filter_type "NN" --decision_delay 2 --total_taps 2 --filter_size 4 --SNR 20 --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 5000 --test_seq_len 50000 --bs 2

# python main.py --epochs 10 --lr 1e-4 --filter_type "NN" --decision_delay 2 --total_taps 2 --filter_size 10 --SNR 20 --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 11000 --test_seq_len 110000 --bs 2
# python main.py --epochs 10 --lr 1e-4 --filter_type "NN" --decision_delay 5 --total_taps 2 --filter_size 10 --SNR 20 --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 11000 --test_seq_len 110000 --bs 2

# python main.py --epochs 10 --lr 1e-4 --filter_type "NN" --decision_delay 5 --total_taps 2 --filter_size 20 --SNR 20 --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 21000 --test_seq_len 210000 --bs 2
# python main.py --epochs 10 --lr 1e-4 --filter_type "NN" --decision_delay 10 --total_taps 2 --filter_size 20 --SNR 20 --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 21000 --test_seq_len 210000 --bs 2

# python main.py --epochs 10 --lr 1e-4 --filter_type "NN" --decision_delay 10 --total_taps 2 --filter_size 40 --SNR 20 --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 41000 --test_seq_len 410000 --bs 2
# python main.py --epochs 10 --lr 1e-4 --filter_type "NN" --decision_delay 20 --total_taps 2 --filter_size 40 --SNR 20 --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 41000 --test_seq_len 410000 --bs 2

# python main.py --epochs 10 --lr 1e-4 --filter_type "NN" --decision_delay 25 --total_taps 2 --filter_size 100 --SNR 20 --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 101000 --test_seq_len 1010000 --bs 2
# python main.py --epochs 10 --lr 1e-4 --filter_type "NN" --decision_delay 50 --total_taps 2 --filter_size 100 --SNR 20 --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 101000 --test_seq_len 1010000 --bs 2

# python main.py --epochs 10 --lr 1e-4 --filter_type "NN" --decision_delay 50 --total_taps 2 --filter_size 200 --SNR 20 --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 201000 --test_seq_len 2010000 --bs 2
# python main.py --epochs 10 --lr 1e-4 --filter_type "NN" --decision_delay 100 --total_taps 2 --filter_size 200 --SNR 20 --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 201000 --test_seq_len 2010000 --bs 2


# python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 2 --filter_size 2 --SNR 20 --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 3000 --test_seq_len 30000 
# python main.py --filter_type "LMMSE" --decision_delay 1 --total_taps 2 --filter_size 2 --SNR 20 --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 3000 --test_seq_len 30000 
# python main.py --filter_type "LMMSE" --decision_delay 2 --total_taps 2 --filter_size 2 --SNR 20 --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 3000 --test_seq_len 30000 

# python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 2 --filter_size 4 --SNR 20 --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 5000 --test_seq_len 50000
# python main.py --filter_type "LMMSE" --decision_delay 1 --total_taps 2 --filter_size 4 --SNR 20 --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 5000 --test_seq_len 50000
# python main.py --filter_type "LMMSE" --decision_delay 2 --total_taps 2 --filter_size 4 --SNR 20 --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 5000 --test_seq_len 50000

# python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 2 --filter_size 10 --SNR 20 --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 11000 --test_seq_len 110000
# python main.py --filter_type "LMMSE" --decision_delay 2 --total_taps 2 --filter_size 10 --SNR 20 --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 11000 --test_seq_len 110000
# python main.py --filter_type "LMMSE" --decision_delay 5 --total_taps 2 --filter_size 10 --SNR 20 --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 11000 --test_seq_len 110000

# python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 2 --filter_size 20 --SNR 20 --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 21000 --test_seq_len 210000
# python main.py --filter_type "LMMSE" --decision_delay 5 --total_taps 2 --filter_size 20 --SNR 20 --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 21000 --test_seq_len 210000
# python main.py --filter_type "LMMSE" --decision_delay 10 --total_taps 2 --filter_size 20 --SNR 20 --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 21000 --test_seq_len 210000

# python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 2 --filter_size 40 --SNR 20 --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 41000 --test_seq_len 410000
# python main.py --filter_type "LMMSE" --decision_delay 10 --total_taps 2 --filter_size 40 --SNR 20 --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 41000 --test_seq_len 410000
# python main.py --filter_type "LMMSE" --decision_delay 20 --total_taps 2 --filter_size 40 --SNR 20 --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 41000 --test_seq_len 410000

# python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 2 --filter_size 100 --SNR 20 --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 101000 --test_seq_len 1010000
# python main.py --filter_type "LMMSE" --decision_delay 25 --total_taps 2 --filter_size 100 --SNR 20 --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 101000 --test_seq_len 1010000
# python main.py --filter_type "LMMSE" --decision_delay 50 --total_taps 2 --filter_size 100 --SNR 20 --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 101000 --test_seq_len 1010000

# python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 2 --filter_size 200 --SNR 20 --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 201000 --test_seq_len 2010000 
# python main.py --filter_type "LMMSE" --decision_delay 50 --total_taps 2 --filter_size 200 --SNR 20 --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 201000 --test_seq_len 2010000 
# python main.py --filter_type "LMMSE" --decision_delay 100 --total_taps 2 --filter_size 200 --SNR 20 --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 201000 --test_seq_len 2010000 


######################### 4 taps #########################
# python main.py --epochs 10 --lr 1e-4 --filter_type "NN" --decision_delay 0 --total_taps 4 --filter_size 4 --SNR 20 --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 7000 --test_seq_len 70000 --bs 2
# python main.py --epochs 10 --lr 1e-4 --filter_type "NN" --decision_delay 0 --total_taps 4 --filter_size 8 --SNR 20 --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 11000 --test_seq_len 110000 --bs 2
# python main.py --epochs 10 --lr 1e-4 --filter_type "NN" --decision_delay 0 --total_taps 4 --filter_size 20 --SNR 20 --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 23000 --test_seq_len 230000 --bs 2
# python main.py --epochs 10 --lr 1e-4 --filter_type "NN" --decision_delay 0 --total_taps 4 --filter_size 40 --SNR 20 --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 43000 --test_seq_len 430000 --bs 2
# python main.py --epochs 10 --lr 1e-4 --filter_type "NN" --decision_delay 0 --total_taps 4 --filter_size 80 --SNR 20 --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 83000 --test_seq_len 830000 --bs 2
# python main.py --epochs 10 --lr 1e-4 --filter_type "NN" --decision_delay 0 --total_taps 4 --filter_size 200 --SNR 20 --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 203000 --test_seq_len 2030000 --bs 2
# python main.py --epochs 10 --lr 1e-4 --filter_type "NN" --decision_delay 0 --total_taps 4 --filter_size 400 --SNR 20 --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 403000 --test_seq_len 4030000 --bs 2


# python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 4 --filter_size 4 --SNR 20 --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 7000 --test_seq_len 70000 
# python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 4 --filter_size 8 --SNR 20 --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 11000 --test_seq_len 110000
# python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 4 --filter_size 20 --SNR 20 --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 23000 --test_seq_len 230000
# python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 4 --filter_size 40 --SNR 20 --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 43000 --test_seq_len 430000
# python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 4 --filter_size 80 --SNR 20 --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 83000 --test_seq_len 830000
# python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 4 --filter_size 200 --SNR 20 --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 203000 --test_seq_len 2030000
# python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 4 --filter_size 400 --SNR 20 --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 403000 --test_seq_len 4030000 


# python main.py --filter_type "LMMSE" --decision_delay 1 --total_taps 4 --filter_size 4 --SNR 20 --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 7000 --test_seq_len 70000 
# python main.py --filter_type "LMMSE" --decision_delay 2 --total_taps 4 --filter_size 8 --SNR 20 --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 11000 --test_seq_len 110000
# python main.py --filter_type "LMMSE" --decision_delay 5 --total_taps 4 --filter_size 20 --SNR 20 --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 23000 --test_seq_len 230000
# python main.py --filter_type "LMMSE" --decision_delay 10 --total_taps 4 --filter_size 40 --SNR 20 --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 43000 --test_seq_len 430000
# python main.py --filter_type "LMMSE" --decision_delay 20 --total_taps 4 --filter_size 80 --SNR 20 --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 83000 --test_seq_len 830000
# python main.py --filter_type "LMMSE" --decision_delay 50 --total_taps 4 --filter_size 200 --SNR 20 --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 203000 --test_seq_len 2030000
# python main.py --filter_type "LMMSE" --decision_delay 100 --total_taps 4 --filter_size 400 --SNR 20 --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 403000 --test_seq_len 4030000 


# python main.py --filter_type "LMMSE" --decision_delay 3 --total_taps 4 --filter_size 4 --SNR 20 --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 7000 --test_seq_len 70000 
# python main.py --filter_type "LMMSE" --decision_delay 5 --total_taps 4 --filter_size 8 --SNR 20 --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 11000 --test_seq_len 110000
# python main.py --filter_type "LMMSE" --decision_delay 11 --total_taps 4 --filter_size 20 --SNR 20 --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 23000 --test_seq_len 230000
# python main.py --filter_type "LMMSE" --decision_delay 21 --total_taps 4 --filter_size 40 --SNR 20 --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 43000 --test_seq_len 430000
# python main.py --filter_type "LMMSE" --decision_delay 41 --total_taps 4 --filter_size 80 --SNR 20 --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 83000 --test_seq_len 830000
# python main.py --filter_type "LMMSE" --decision_delay 101 --total_taps 4 --filter_size 200 --SNR 20 --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 203000 --test_seq_len 2030000
# python main.py --filter_type "LMMSE" --decision_delay 201 --total_taps 4 --filter_size 400 --SNR 20 --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 403000 --test_seq_len 4030000 



######################### 8 taps #########################

python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 1500000 --test_seq_len 1500000
python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 16 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 2300000 --test_seq_len 2300000
python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 40 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 4700000 --test_seq_len 4700000
python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 80 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 8700000 --test_seq_len 8700000
python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 160 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 16700000 --test_seq_len 16700000
python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 400 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 40700000 --test_seq_len 40700000
python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 800 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 80700000 --test_seq_len 80700000

python main.py --epochs 300 --lr 1e-8 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 1500000 --test_seq_len 1500000 --bs 1024
python main.py --epochs 300 --lr 1e-8 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 16 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 2300000 --test_seq_len 2300000 --bs 1024
python main.py --epochs 300 --lr 1e-8 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 40 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 4700000 --test_seq_len 4700000 --bs 1024
python main.py --epochs 300 --lr 1e-8 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 80 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 8700000 --test_seq_len 8700000 --bs 1024
python main.py --epochs 300 --lr 1e-8 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 160 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 16700000 --test_seq_len 16700000 --bs 1024
python main.py --epochs 300 --lr 1e-8 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 400 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 40700000 --test_seq_len 40700000 --bs 1024
python main.py --epochs 300 --lr 1e-8 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 800 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 80700000 --test_seq_len 80700000 --bs 1024



python main.py --filter_type "LMMSE" --decision_delay 3 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 1500000 --test_seq_len 1500000
python main.py --filter_type "LMMSE" --decision_delay 5 --total_taps 8 --filter_size 16 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 2300000 --test_seq_len 2300000
python main.py --filter_type "LMMSE" --decision_delay 11 --total_taps 8 --filter_size 40 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 4700000 --test_seq_len 4700000
python main.py --filter_type "LMMSE" --decision_delay 21 --total_taps 8 --filter_size 80 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 8700000 --test_seq_len 8700000
python main.py --filter_type "LMMSE" --decision_delay 41 --total_taps 8 --filter_size 160 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 16700000 --test_seq_len 16700000
python main.py --filter_type "LMMSE" --decision_delay 101 --total_taps 8 --filter_size 400 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 40700000 --test_seq_len 40700000
python main.py --filter_type "LMMSE" --decision_delay 201 --total_taps 8 --filter_size 800 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 80700000 --test_seq_len 80700000

python main.py --epochs 300 --lr 1e-8 --filter_type "NN" --decision_delay 3 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 1500000 --test_seq_len 1500000 --bs 1024
python main.py --epochs 300 --lr 1e-8 --filter_type "NN" --decision_delay 5 --total_taps 8 --filter_size 16 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 2300000 --test_seq_len 2300000 --bs 1024
python main.py --epochs 300 --lr 1e-8 --filter_type "NN" --decision_delay 11 --total_taps 8 --filter_size 40 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 4700000 --test_seq_len 4700000 --bs 1024
python main.py --epochs 300 --lr 1e-8 --filter_type "NN" --decision_delay 21 --total_taps 8 --filter_size 80 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 8700000 --test_seq_len 8700000 --bs 1024
python main.py --epochs 300 --lr 1e-8 --filter_type "NN" --decision_delay 41 --total_taps 8 --filter_size 160 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 16700000 --test_seq_len 16700000 --bs 1024
python main.py --epochs 300 --lr 1e-8 --filter_type "NN" --decision_delay 101 --total_taps 8 --filter_size 400 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 40700000 --test_seq_len 40700000 --bs 1024
python main.py --epochs 300 --lr 1e-8 --filter_type "NN" --decision_delay 201 --total_taps 8 --filter_size 800 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 80700000 --test_seq_len 80700000 --bs 1024



python main.py --filter_type "LMMSE" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 1500000 --test_seq_len 1500000
python main.py --filter_type "LMMSE" --decision_delay 11 --total_taps 8 --filter_size 16 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 2300000 --test_seq_len 2300000
python main.py --filter_type "LMMSE" --decision_delay 23 --total_taps 8 --filter_size 40 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 4700000 --test_seq_len 4700000
python main.py --filter_type "LMMSE" --decision_delay 43 --total_taps 8 --filter_size 80 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 8700000 --test_seq_len 8700000
python main.py --filter_type "LMMSE" --decision_delay 83 --total_taps 8 --filter_size 160 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 16700000 --test_seq_len 16700000
python main.py --filter_type "LMMSE" --decision_delay 203 --total_taps 8 --filter_size 400 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 40700000 --test_seq_len 40700000
python main.py --filter_type "LMMSE" --decision_delay 403 --total_taps 8 --filter_size 800 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 80700000 --test_seq_len 80700000

python main.py --epochs 300 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 1500000 --test_seq_len 1500000 --bs 1024
python main.py --epochs 300 --lr 1e-8 --filter_type "NN" --decision_delay 11 --total_taps 8 --filter_size 16 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 2300000 --test_seq_len 2300000 --bs 1024
python main.py --epochs 300 --lr 1e-8 --filter_type "NN" --decision_delay 23 --total_taps 8 --filter_size 40 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 4700000 --test_seq_len 4700000 --bs 1024
python main.py --epochs 300 --lr 1e-8 --filter_type "NN" --decision_delay 43 --total_taps 8 --filter_size 80 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 8700000 --test_seq_len 8700000 --bs 1024
python main.py --epochs 300 --lr 1e-8 --filter_type "NN" --decision_delay 83 --total_taps 8 --filter_size 160 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 16700000 --test_seq_len 16700000 --bs 1024
python main.py --epochs 300 --lr 1e-8 --filter_type "NN" --decision_delay 203 --total_taps 8 --filter_size 400 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 40700000 --test_seq_len 40700000 --bs 1024
python main.py --epochs 300 --lr 1e-8 --filter_type "NN" --decision_delay 403 --total_taps 8 --filter_size 800 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 80700000 --test_seq_len 80700000 --bs 1024


###################
# python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 80 --SNR 0 --rand_seed_channel 5551 --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 8700000 --test_seq_len 8700000
# python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 80 --SNR 10 --rand_seed_channel 5551 --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 8700000 --test_seq_len 8700000
# python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 80 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 8700000 --test_seq_len 8700000
# python main.py --epochs 300 --lr 1e-8 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 80 --SNR 0 --rand_seed_channel 5551 --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 8700000 --test_seq_len 8700000 --bs 1024
# python main.py --epochs 300 --lr 1e-8 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 80 --SNR 10 --rand_seed_channel 5551 --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 8700000 --test_seq_len 8700000 --bs 1024
# python main.py --epochs 300 --lr 1e-8 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 80 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 8700000 --test_seq_len 8700000 --bs 1024


# python main.py --filter_type "LMMSE" --decision_delay 21 --total_taps 8 --filter_size 80 --SNR 0 --rand_seed_channel 5551 --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 8700000 --test_seq_len 8700000
# python main.py --filter_type "LMMSE" --decision_delay 21 --total_taps 8 --filter_size 80 --SNR 10 --rand_seed_channel 5551 --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 8700000 --test_seq_len 8700000
# python main.py --filter_type "LMMSE" --decision_delay 21 --total_taps 8 --filter_size 80 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 8700000 --test_seq_len 8700000
# python main.py --epochs 300 --lr 1e-8 --filter_type "NN" --decision_delay 21 --total_taps 8 --filter_size 80 --SNR 0 --rand_seed_channel 5551 --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 8700000 --test_seq_len 8700000 --bs 1024
# python main.py --epochs 300 --lr 1e-8 --filter_type "NN" --decision_delay 21 --total_taps 8 --filter_size 80 --SNR 10 --rand_seed_channel 5551 --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 8700000 --test_seq_len 8700000 --bs 1024
# python main.py --epochs 300 --lr 1e-8 --filter_type "NN" --decision_delay 21 --total_taps 8 --filter_size 80 --SNR 20 --rand_seed_channel 5551 --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 8700000 --test_seq_len 8700000 --bs 1024
###################


#python main.py --epochs 1000 --lr 1e-8 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 80 --SNR 20 --rand_seed_channel 4572 --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 8700000 --test_seq_len 8700000 --bs 4096

#python main.py --epochs 800 --lr 1e-6 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 80 --SNR 30 --rand_seed_channel 7245 --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 8700000 --test_seq_len 8700000 --bs 1024


#python main.py --epochs 500 --lr 1e-6 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 80 --SNR -20 --rand_seed_channel 1211 --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 1740000 --test_seq_len 870000 --bs 512
#python main.py --epochs 500 --lr 1e-6 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 80 --SNR 0 --rand_seed_channel 1211 --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 1740000 --test_seq_len 870000 --bs 512
#python main.py --epochs 800 --lr 1e-6 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 80 --SNR -10 --rand_seed_channel 4375 --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 8700000 --test_seq_len 8700000 --bs 1024
#python main.py --epochs 800 --lr 1e-6 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 80 --SNR 30 --rand_seed_channel 4375 --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 8700000 --test_seq_len 8700000 --bs 1024


# python main.py --epochs 440 --lr 1e-5 --filter_type "NN" --decision_delay 5 --total_taps 8 --filter_size 16 --SNR -20 --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 23000 --test_seq_len 230000 --bs 256
# python main.py --epochs 440 --lr 1e-5 --filter_type "NN" --decision_delay 5 --total_taps 8 --filter_size 16 --SNR 0 --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 23000 --test_seq_len 230000 --bs 256
# python main.py --epochs 440 --lr 1e-5 --filter_type "NN" --decision_delay 5 --total_taps 8 --filter_size 16 --SNR 20 --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 23000 --test_seq_len 230000 --bs 256
# python main.py --epochs 20 --lr 1e-4 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 160 --SNR 20 --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 334000 --test_seq_len 1670000 --bs 8
# python main.py --epochs 20 --lr 1e-4 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 400 --SNR 20 --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 814000 --test_seq_len 4070000 --bs 8
# python main.py --epochs 20 --lr 1e-4 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 800 --SNR 20 --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 1614000 --test_seq_len 8070000 --bs 8


# python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 15000 --test_seq_len 150000
# python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 16 --SNR 20 --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 23000 --test_seq_len 230000
# python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 40 --SNR 20 --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 47000 --test_seq_len 470000


# python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 80 --SNR -20 --rand_seed_channel 3451 --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 87000 --test_seq_len 870000
# python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 80 --SNR 0 --rand_seed_channel 3451 --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 87000 --test_seq_len 870000

#python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 80 --SNR -10 --rand_seed_channel 7245 --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 3480000 --test_seq_len 8700000
#python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 80 --SNR 0 --rand_seed_channel 7245 --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 3480000 --test_seq_len 8700000
#python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 80 --SNR 10 --rand_seed_channel 7245 --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 3480000 --test_seq_len 8700000
#python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 80 --SNR 20 --rand_seed_channel 7245 --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 3480000 --test_seq_len 8700000
#python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 80 --SNR 30 --rand_seed_channel 7245 --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 3480000 --test_seq_len 8700000

# python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 80 --SNR -20 --rand_seed_channel 1211 --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 87000 --test_seq_len 870000
# python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 80 --SNR 0 --rand_seed_channel 1211 --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 87000 --test_seq_len 870000
#python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 80 --SNR 30 --rand_seed_channel 4375 --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 3480000 --test_seq_len 8700000

