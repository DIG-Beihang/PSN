 GPU_ID=1
 for((iter=100000;iter>=30000;iter-=5000));
 do
     echo $iter;
     CUDA_VISIBLE_DEVICES=$GPU_ID python test.py \
                         --dataset cityscape_fog --net vgg16 \
                         --cuda --load_name models/vgg16/cityscape_elasticgcn_final_fc_domain_5e-5lr_01gp/target_cityscape_fog_step_$iter.pth --bs 1 --nw 0
 done
