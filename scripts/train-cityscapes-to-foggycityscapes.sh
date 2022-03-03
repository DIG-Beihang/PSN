GPU_ID=0
CUDA_VISIBLE_DEVICES=$GPU_ID python -u main.py \
                        --dataset cityscape --dataset_t foggy_cityscape --net vgg16 \
                        --cuda --bs 1 --nw 0 --warmup 10000 --out '' --lr 0.001 \
			--model_c fc 2>&1 | tee train_log/train-cityscape-to-foggy_cityscape.log
