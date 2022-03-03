 GPU_ID=0
 for((iter=70000;iter>=30000;iter-=5000));
 do
     echo $iter;
     CUDA_VISIBLE_DEVICES=$GPU_ID python test-2.py \
                         --dataset domain2 --net vgg16 \
                         --cuda --load_name models/vgg16/domain1gcn_final_fc_domain_5e-5lr_01gp/target_domain2_step_$iter.pth --bs 1 --nw 0
 done
