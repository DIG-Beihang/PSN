 GPU_ID=0
 for((iter=70000;iter>=30000;iter-=5000));
 do
     echo $iter;
     CUDA_VISIBLE_DEVICES=$GPU_ID python test.py \
                         --dataset foggy_cityscape --net vgg16 \
                         --cuda --load_name models/vgg16/cityscape/target_foggy_cityscape_step_$iter.pth --bs 1 --nw 0
 done
