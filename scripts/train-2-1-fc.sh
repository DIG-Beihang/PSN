GPU_ID=0
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
                        --dataset domain2 --dataset_t domain1 --net vgg16 \
                        --cuda --bs 1 --nw 0 --warmup 10000 --out gcn_final_fc_domain_5e-5lr_01gp \
			--model_c fc --log_dir train_log
