GPU_ID=1
CUDA_VISIBLE_DEVICES=$GPU_ID python -u main.py \
                        --dataset cityscape_den --dataset_t cityscape_res --net vgg16 \
                        --cuda --bs 1 --nw 0 --warmup 10000 --out gcn_final_fc_domain_5e-5lr_01gp \
			--model_c fc --step 100000 2>&1 | tee train_log/train-den-res.log
