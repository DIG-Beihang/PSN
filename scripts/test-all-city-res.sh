 GPU_ID=2
 for((iter=70000;iter<=100000;iter+=5000));
 do
     echo $iter;
     CUDA_VISIBLE_DEVICES=$GPU_ID python test.py \
                         --dataset cityscape_res --net vgg16 \
                         --cuda --load_name models/vgg16/cityscapegcn_final_fc_domain_5e-5lr_01gp/target_cityscape_res_step_$iter.pth --bs 1 --nw 0
 done
