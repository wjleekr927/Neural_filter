# Start simulation
# Control SNR and decision_delay by bash variable

# First variable can be used as $1.
# e.g.) SNR=$1

######################### 8 taps #########################
# Drop 2, 9 th seed
# 2169 5551 6438 0405 3438 2077 0927 7522 9716 8837
# 2169 6438 0405 3438 2077 0927 7522 8837 (for 8 averaging)

# # First exp (M/N_c)

# for idx in 927	7758	3722	9017	9570	7908	8422	8327	6043	7757	4604	8086	3697	5292	3790	4122	8850	8013	5554	9675	3714	6346	5973	8473	8666	4706	6678	6390	7697	8107	8403	5286	7914	7754	4548	4264	6734	9737	5705	7300	4947	8380	5150	6783	8040	9289	9734	7052	4392	4461	5166	8963	5145	8790	5075	9539	5768	4769	5124	7500	6571	5779	8898	7300	7068	9460	5350	8419	8396	5966	7186	3983	3841	6945	8562	9570	4335	7193	6545	3567	5684	4545	8660	5516	6930	4568	7408	5201	7748	7976	8360	6423	4035	4980	9435	4481	8866	405	2169	3438
for idx in 6438 3438 2169 
do
        python main.py --filter_type "Viterbi" --decision_delay 2 --total_taps 8 --filter_size 8 --SNR 15 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 3000
# #     # Tau = 0
# #     python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000
# #     python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 16 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 460000 --test_seq_len 13800000
# #     python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 40 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 940000 --test_seq_len 28200000
# #     python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 80 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 1740000 --test_seq_len 52200000
# #     python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 160 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 3340000 --test_seq_len 100200000
# #     python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 400 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 8140000 --test_seq_len 244200000
# #     python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 800 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 16140000 --test_seq_len 484200000 

# #     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 64
# #     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 16 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 460000 --test_seq_len 13800000 --bs 64
# #     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 40 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 940000 --test_seq_len 28200000 --bs 64
# #     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 80 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 1740000 --test_seq_len 52200000 --bs 64
# #     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 160 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 3340000 --test_seq_len 100200000 --bs 64
# #     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 400 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 8140000 --test_seq_len 244200000 --bs 64
# #     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 800 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 16140000 --test_seq_len 484200000 --bs 64

# #     # Tau = L // 4
# #     python main.py --filter_type "LMMSE" --decision_delay 3 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000
# #     python main.py --filter_type "LMMSE" --decision_delay 5 --total_taps 8 --filter_size 16 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 460000 --test_seq_len 13800000
# #     python main.py --filter_type "LMMSE" --decision_delay 11 --total_taps 8 --filter_size 40 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 940000 --test_seq_len 28200000
# #     python main.py --filter_type "LMMSE" --decision_delay 21 --total_taps 8 --filter_size 80 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 1740000 --test_seq_len 52200000
# #     python main.py --filter_type "LMMSE" --decision_delay 41 --total_taps 8 --filter_size 160 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 3340000 --test_seq_len 100200000
# #     python main.py --filter_type "LMMSE" --decision_delay 101 --total_taps 8 --filter_size 400 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 8140000 --test_seq_len 244200000
# #     python main.py --filter_type "LMMSE" --decision_delay 201 --total_taps 8 --filter_size 800 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 16140000 --test_seq_len 484200000 

# #     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 3 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 64
# #     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 5 --total_taps 8 --filter_size 16 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 460000 --test_seq_len 13800000 --bs 64
# #     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 11 --total_taps 8 --filter_size 40 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 940000 --test_seq_len 28200000 --bs 64
# #     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 21 --total_taps 8 --filter_size 80 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 1740000 --test_seq_len 52200000 --bs 64
# #     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 41 --total_taps 8 --filter_size 160 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 3340000 --test_seq_len 100200000 --bs 64
# #     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 101 --total_taps 8 --filter_size 400 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 8140000 --test_seq_len 244200000 --bs 64
# #     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 201 --total_taps 8 --filter_size 800 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 16140000 --test_seq_len 484200000 --bs 64

    # Tau = L // 2
    #python main.py --filter_type "LMMSE" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 1
    #python main.py --filter_type "LMMSE" --decision_delay 11 --total_taps 8 --filter_size 16 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 460000 --test_seq_len 13800000 --exp_num 5
    #python main.py --filter_type "LMMSE" --decision_delay 23 --total_taps 8 --filter_size 40 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 940000 --test_seq_len 28200000 --exp_num 5
    #python main.py --filter_type "LMMSE" --decision_delay 43 --total_taps 8 --filter_size 80 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 1740000 --test_seq_len 52200000 --exp_num 5
    #python main.py --filter_type "LMMSE" --decision_delay 83 --total_taps 8 --filter_size 160 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 3340000 --test_seq_len 100200000 --exp_num 1
    #python main.py --filter_type "LMMSE" --decision_delay 203 --total_taps 8 --filter_size 400 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 8140000 --test_seq_len 244200000 --exp_num 1
    #python main.py --filter_type "LMMSE" --decision_delay 403 --total_taps 8 --filter_size 800 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 16140000 --test_seq_len 484200000 --exp_num 1

    #python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 128 --exp_num 1 --gpu 0
    #python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 11 --total_taps 8 --filter_size 16 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9002 --rand_seed_test 4002 --train_seq_len 460000 --test_seq_len 13800000 --bs 128 --exp_num 5 --gpu 0
    #python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 23 --total_taps 8 --filter_size 40 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9005 --rand_seed_test 4005 --train_seq_len 940000 --test_seq_len 28200000 --bs 128 --exp_num 5 --gpu 0
    #python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 43 --total_taps 8 --filter_size 80 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9010 --rand_seed_test 4010 --train_seq_len 1740000 --test_seq_len 52200000 --bs 128 --exp_num 5 --gpu 0
    #python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 83 --total_taps 8 --filter_size 160 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9020 --rand_seed_test 4020 --train_seq_len 3340000 --test_seq_len 100200000 --bs 128 --exp_num 1 --gpu 0
    #python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 203 --total_taps 8 --filter_size 400 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9050 --rand_seed_test 4050 --train_seq_len 8140000 --test_seq_len 244200000 --bs 128 --exp_num 1 --gpu 0
    #python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 403 --total_taps 8 --filter_size 800 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9100 --rand_seed_test 4100 --train_seq_len 16140000 --test_seq_len 484200000 --bs 128 --exp_num 1 --gpu 0

done


# # Second exp (Training data)

# for idx in 927	7758	3722	9017	9570	7908	8422	8327	6043	7757	4604	8086	3697	5292	3790	4122	8850	8013	5554	9675	3714	6346	5973	8473	8666	4706	6678	6390	7697	8107	8403	5286	7914	7754	4548	4264	6734	9737	5705	7300	4947	8380	5150	6783	8040	9289	9734	7052	4392	4461	5166	8963	5145	8790	5075	9539	5768	4769	5124	7500	6571	5779	8898	7300	7068	9460	5350	8419	8396	5966	7186	3983	3841	6945	8562	9570	4335	7193	6545	3567	5684	4545	8660	5516	6930	4568	7408	5201	7748	7976	8360	6423	4035	4980	9435	4481	8866	405	2169	3438
# do
#     python main.py --filter_type "LMMSE" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 150000 --test_seq_len 9000000 --exp_num 2 --gpu 0 # --scatter_plot

#     python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 30 --test_seq_len 9000000 --exp_num 2 --gpu 0
#     python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 60 --test_seq_len 9000000 --exp_num 2 --gpu 0
#     python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300 --test_seq_len 9000000 --exp_num 2 --gpu 0
#     python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 600 --test_seq_len 9000000 --exp_num 2 --gpu 0
#     python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 3000 --test_seq_len 9000000 --exp_num 2 --gpu 0
#     python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 6000 --test_seq_len 9000000 --exp_num 2 --gpu 0
#     python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 30000 --test_seq_len 9000000 --exp_num 2 --gpu 0
#     python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 60000 --test_seq_len 9000000 --exp_num 2 --gpu 0
#     python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 150000 --test_seq_len 9000000 --exp_num 2 --gpu 0 # --scatter_plot
#     python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 2 --gpu 0

#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 30 --test_seq_len 9000000 --bs 1 --exp_num 2 --gpu 0
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 60 --test_seq_len 9000000 --bs 2 --exp_num 2 --gpu 0
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300 --test_seq_len 9000000 --bs 4 --exp_num 2 --gpu 0
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 600 --test_seq_len 9000000 --bs 8 --exp_num 2 --gpu 0
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 3000 --test_seq_len 9000000 --bs 8 --exp_num 2 --gpu 0
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 6000 --test_seq_len 9000000 --bs 16 --exp_num 2 --gpu 0
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 30000 --test_seq_len 9000000 --bs 16 --exp_num 2 --gpu 0
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 60000 --test_seq_len 9000000 --bs 32 --exp_num 2 --gpu 0
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 150000 --test_seq_len 9000000 --bs 32 --exp_num 2 --gpu 0 # --scatter_plot
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 128 --exp_num 2 --gpu 0 # --scatter_plot

# done


# # Third exp (SNR)

# for SNR in -10 -5 0 5 10 15 20 25 30 35 40
# do
#     for idx in 927	7758	3722	9017	9570	7908	8422	8327	6043	7757	4604	8086	3697	5292	3790	4122	8850	8013	5554	9675	3714	6346	5973	8473	8666	4706	6678	6390	7697	8107	8403	5286	7914	7754	4548	4264	6734	9737	5705	7300	4947	8380	5150	6783	8040	9289	9734	7052	4392	4461	5166	8963	5145	8790	5075	9539	5768	4769	5124	7500	6571	5779	8898	7300	7068	9460	5350	8419	8396	5966	7186	3983	3841	6945	8562	9570	4335	7193	6545	3567	5684	4545	8660	5516	6930	4568	7408	5201	7748	7976	8360	6423	4035	4980	9435	4481	8866	405	2169	3438
#     do
#         python main.py --filter_type "LMMSE" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 150000 --test_seq_len 9000000 --exp_num 3 # --scatter_plot

#         # python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 30 --test_seq_len 9000000
#         # python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 60 --test_seq_len 9000000
#         # python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300 --test_seq_len 9000000
#         # python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 600 --test_seq_len 9000000 
#         # python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 3000 --test_seq_len 9000000
#         # python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 6000 --test_seq_len 9000000
#         # python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 30000 --test_seq_len 9000000
#         # python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 60000 --test_seq_len 9000000
#         # python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 150000 --test_seq_len 9000000 # --scatter_plot
#         python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 3

#         # python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 30 --test_seq_len 9000000 --bs 1
#         # python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 60 --test_seq_len 9000000 --bs 2
#         # python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300 --test_seq_len 9000000 --bs 4
#         # python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 600 --test_seq_len 9000000 --bs 8
#         # python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 3000 --test_seq_len 9000000 --bs 8
#         # python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 6000 --test_seq_len 9000000 --bs 16
#         # python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 30000 --test_seq_len 9000000 --bs 16
#         # python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 60000 --test_seq_len 9000000 --bs 32
#         # python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 150000 --test_seq_len 9000000 --bs 32 --scatter_plot
#         python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR $SNR --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 128 --exp_num 3 # --scatter_plot
#     done
# done




# # Fourth exp (tau)

# for idx in 927	7758	3722	9017	9570	7908	8422	8327	6043	7757	4604	8086	3697	5292	3790	4122	8850	8013	5554	9675	3714	6346	5973	8473	8666	4706	6678	6390	7697	8107	8403	5286	7914	7754	4548	4264	6734	9737	5705	7300	4947	8380	5150	6783	8040	9289	9734	7052	4392	4461	5166	8963	5145	8790	5075	9539	5768	4769	5124	7500	6571	5779	8898	7300	7068	9460	5350	8419	8396	5966	7186	3983	3841	6945	8562	9570	4335	7193	6545	3567	5684	4545	8660	5516	6930	4568	7408	5201	7748	7976	8360	6423	4035	4980	9435	4481	8866	405	2169	3438
# do
#     # Tau = 0
#     python main.py --filter_type "LMMSE" --decision_delay 0 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 4
#     python main.py --filter_type "LS" --decision_delay 0 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 4
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 0 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 128 --exp_num 4

#     python main.py --filter_type "LMMSE" --decision_delay 1 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 4
#     python main.py --filter_type "LS" --decision_delay 1 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 4
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 1 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 128 --exp_num 4

#     python main.py --filter_type "LMMSE" --decision_delay 2 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 4
#     python main.py --filter_type "LS" --decision_delay 2 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 4
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 2 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 128 --exp_num 4

#     # Tau = L // 4
#     python main.py --filter_type "LMMSE" --decision_delay 3 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 4
#     python main.py --filter_type "LS" --decision_delay 3 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 4
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 3 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 128 --exp_num 4

#     python main.py --filter_type "LMMSE" --decision_delay 4 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 4
#     python main.py --filter_type "LS" --decision_delay 4 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 4
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 4 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 128 --exp_num 4

#     python main.py --filter_type "LMMSE" --decision_delay 5 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 4
#     python main.py --filter_type "LS" --decision_delay 5 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 4
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 5 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 128 --exp_num 4

#     python main.py --filter_type "LMMSE" --decision_delay 6 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 4
#     python main.py --filter_type "LS" --decision_delay 6 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 4
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 6 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 128 --exp_num 4

#     # Tau = L // 2
#     python main.py --filter_type "LMMSE" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 4
#     python main.py --filter_type "LS" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 4
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 7 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 128 --exp_num 4

#     python main.py --filter_type "LMMSE" --decision_delay 8 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 4
#     python main.py --filter_type "LS" --decision_delay 8 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 4
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 8 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 128 --exp_num 4

#     python main.py --filter_type "LMMSE" --decision_delay 9 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 4
#     python main.py --filter_type "LS" --decision_delay 9 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 4
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 9 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 128 --exp_num 4
    
#     python main.py --filter_type "LMMSE" --decision_delay 10 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 4
#     python main.py --filter_type "LS" --decision_delay 10 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 4
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 10 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 128 --exp_num 4

#     python main.py --filter_type "LMMSE" --decision_delay 11 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 4
#     python main.py --filter_type "LS" --decision_delay 11 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 4
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 11 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 128 --exp_num 4

#     python main.py --filter_type "LMMSE" --decision_delay 12 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 4
#     python main.py --filter_type "LS" --decision_delay 12 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 4
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 12 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 128 --exp_num 4

#     python main.py --filter_type "LMMSE" --decision_delay 13 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 4
#     python main.py --filter_type "LS" --decision_delay 13 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 4
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 13 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 128 --exp_num 4

#     python main.py --filter_type "LMMSE" --decision_delay 14 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 4
#     python main.py --filter_type "LS" --decision_delay 14 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --exp_num 4
#     python main.py --epochs 360 --lr 1e-8 --filter_type "NN" --decision_delay 14 --total_taps 8 --filter_size 8 --SNR 20 --rand_seed_channel $idx --rand_seed_train 9001 --rand_seed_test 4001 --train_seq_len 300000 --test_seq_len 9000000 --bs 128 --exp_num 4

# done