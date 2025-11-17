export CUDA_VISIBLE_DEVICES=6

# python launch.py  --mode=train \
#     --train_dataset="10_000 @ PointOdysseyDUSt3R(dset='train', z_far=80, dataset_location='data/point_odyssey', S=2, aug_crop=16, resolution=[(512, 288), (512, 384), (512, 336)], transform=ColorJitter, strides=[1,2,3,4,5,6,7,8,9], dist_type='linear_1_2', aug_focal=0.9)+ 5_000 @ TarTanAirDUSt3R(dset='Hard', z_far=80, dataset_location='data/tartanair', S=2, aug_crop=16, resolution=[(512, 288), (512, 384), (512, 336)], transform=ColorJitter, strides=[1,2,3,4,5,6,7,8,9], dist_type='linear_1_2', aug_focal=0.9)+ 1_000 @ SpringDUSt3R(dset='train', z_far=80, dataset_location='data/spring', S=2, aug_crop=16, resolution=[(512, 288), (512, 384), (512, 336)], transform=ColorJitter, strides=[1,2,3,4,5,6,7,8,9], dist_type='linear_1_2', aug_focal=0.9)+ 4_000 @ Waymo(ROOT='data/waymo_processed', pairs_npz_name='waymo_pairs_video.npz', aug_crop=16, resolution=[(512, 288), (512, 384), (512, 336)], transform=ColorJitter, aug_focal=0.9)"   \
#     --test_dataset="1000 @ PointOdysseyDUSt3R(dset='test', z_far=80, dataset_location='data/point_odyssey', S=2, strides=[1,2,3,4,5,6,7,8,9], resolution=[(512, 288)], seed=777)+ 1000 @ SintelDUSt3R(dset='final', z_far=80, S=2, strides=[1,2,3,4,5,6,7,8,9], resolution=[(512, 224)], seed=777)"   \
#     --train_criterion="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)"  \
#     --test_criterion="Regr3D_ScaleShiftInv(L21, gt_scale=True)"   \
#     --pretrained="checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"   \
#     --lr=0.00005 --min_lr=1e-06 --warmup_epochs=3 --epochs=50 --batch_size=4 --accum_iter=4  \
#     --save_freq=3 --keep_freq=5 --eval_freq=1  \
#     --output_dir="results/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt"

# training only on PointOdyssey to see how well it does
# python launch.py  --mode=train \
#     --train_dataset="10_000 @ PointOdysseyDUSt3R(dset='train', z_far=80, dataset_location='data/point_odyssey', S=2, aug_crop=16, resolution=[(512, 288), (512, 384), (512, 336)], transform=ColorJitter, strides=[1,2,3,4,5,6,7,8,9], dist_type='linear_1_2', aug_focal=0.9)"   \
#     --test_dataset="1000 @ PointOdysseyDUSt3R(dset='test', z_far=80, dataset_location='data/point_odyssey', S=2, strides=[1,2,3,4,5,6,7,8,9], resolution=[(512, 288)], seed=777)"   \
#     --train_criterion="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)"  \
#     --test_criterion="Regr3D_ScaleShiftInv(L21, gt_scale=True)"   \
#     --pretrained="checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"   \
#     --lr=0.00005 --min_lr=1e-06 --warmup_epochs=3 --epochs=1 --batch_size=2 --accum_iter=4  \
#     --save_freq=3 --keep_freq=5 --eval_freq=1  \
#     --output_dir="results/MonST3R_PO_ViTLarge_BaseDecoder_512_dpt-test"



# evaluaton on DAVIS
# python launch.py --mode=eval_pose  \
#     --pretrained="results/MonST3R_PO_ViTLarge_BaseDecoder_512_dpt-test/checkpoint-final.pth"   \
#     --eval_dataset=davis --output_dir="results/davis_joint" 
#     # To use the ground truth dynamic mask for davis, add: --use_gt_mask



# training only on EventPointOdyssey to see how well it does
# python launch.py  --mode=train \
#     --train_dataset="10_000 @ EventPointOdysseyDUSt3R(dset='train', z_far=80, dataset_location='data/point_odyssey', S=2, aug_crop=16, resolution=[(512, 288), (512, 384), (512, 336)], transform=ColorJitter, strides=[1,2,3,4,5,6,7,8,9], dist_type='linear_1_2', aug_focal=0.9)"   \
#     --test_dataset="1000 @ EventPointOdysseyDUSt3R(dset='test', z_far=80, dataset_location='data/point_odyssey', S=2, strides=[1,2,3,4,5,6,7,8,9], resolution=[(512, 288)], seed=777)"   \
#     --model="AsymmetricCroCo3DEventStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, freeze='encoder')"  \
#     --train_criterion="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)"  \
#     --test_criterion="Regr3D_ScaleShiftInv(L21, gt_scale=True)"   \
#     --lr=0.00005 --min_lr=1e-06 --warmup_epochs=3 --epochs=50 --batch_size=2 --accum_iter=4  \
#     --save_freq=3 --keep_freq=5 --eval_freq=1  \
#     --output_dir="results/MonST3R_PO_Event" > results/MonST3R_PO_Event.txt 2>&1


# training only on PointOdyssey to see how well it does
# python launch.py  --mode=train \
#     --train_dataset="10_000 @ PointOdysseyDUSt3R(dset='train', z_far=80, dataset_location='data/point_odyssey', S=2, aug_crop=16, resolution=[(512, 288), (512, 384), (512, 336)], transform=ColorJitter, strides=[1,2,3,4,5,6,7,8,9], dist_type='linear_1_2', aug_focal=0.9)"   \
#     --test_dataset="1000 @ PointOdysseyDUSt3R(dset='test', z_far=80, dataset_location='data/point_odyssey', S=2, strides=[1,2,3,4,5,6,7,8,9], resolution=[(512, 288)], seed=777)"   \
#     --model="AsymmetricCroCo3DStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, freeze='encoder')"  \
#     --train_criterion="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)"  \
#     --test_criterion="Regr3D_ScaleShiftInv(L21, gt_scale=True)"   \
#     --lr=0.00005 --min_lr=1e-06 --warmup_epochs=3 --epochs=50 --batch_size=2 --accum_iter=4  \
#     --save_freq=3 --keep_freq=5 --eval_freq=1  \
#     --output_dir="results/MonST3R_PO_RGB" > results/MonST3R_PO_RGB.txt 2>&1

#training only on EventRGBPointOdyssey to see how well it does
# python launch.py  --mode=train \
#     --train_dataset="10_000 @ EventPointOdysseyDUSt3R(dset='train', z_far=80, dataset_location='data/point_odyssey', S=2, aug_crop=16, resolution=[(512, 288), (512, 384), (512, 336)], transform=ColorJitter, strides=[1,2,3,4,5,6,7,8,9], dist_type='linear_1_2', aug_focal=0.9)"   \
#     --test_dataset="1000 @ EventPointOdysseyDUSt3R(dset='test', z_far=80, dataset_location='data/point_odyssey', S=2, strides=[1,2,3,4,5,6,7,8,9], resolution=[(512, 288)], seed=777)"   \
#     --model="AsymmetricCroCo3DEventRGBStereoV2(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, freeze='encoder')"  \
#     --train_criterion="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)"  \
#     --test_criterion="Regr3D_ScaleShiftInv(L21, gt_scale=True)"   \
#     --lr=0.00005 --min_lr=1e-06 --warmup_epochs=3 --epochs=50 --batch_size=2 --accum_iter=4  \
#     --save_freq=3 --keep_freq=5 --eval_freq=1  \
#     --output_dir="results/MonST3R_PO_Event_RGBV2_test" #> results/MonST3R_PO_Event_RGBV2.txt 2>&1

# VGGT_v1 EventRGB training on PointOdyssey
python launch.py  --mode=train \
    --train_dataset="10_000 @ EventPointOdysseyDUSt3R(dset='train', z_far=80, dataset_location='data/point_odyssey', S=2, aug_crop=16, resolution=[(512, 288), (512, 384), (512, 336)], transform=ColorJitter, strides=[1,2,3,4,5,6,7,8,9], dist_type='linear_1_2', aug_focal=0.9)"   \
    --test_dataset="1000 @ EventPointOdysseyDUSt3R(dset='test', z_far=80, dataset_location='data/point_odyssey', S=2, strides=[1,2,3,4,5,6,7,8,9], resolution=[(512, 288)], seed=777)"   \
    --model="VGGTEventRGBStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=2048, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, freeze='encoder')"  \
    --train_criterion="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)"  \
    --test_criterion="Regr3D_ScaleShiftInv(L21, gt_scale=True)"   \
    --lr=0.00005 --min_lr=1e-06 --warmup_epochs=3 --epochs=50 --batch_size=2 --accum_iter=4  \
    --save_freq=3 --keep_freq=5 --eval_freq=1  \
    --pretrained="/storage/dulanga/4DRecon/Event_STream3R/data/4DRecon/pretrained/vggt.pt"   \
    --output_dir="results2/MonST3R_PO_Event_RGB_VGGT_pretrained_no_ev_norm" > results2/MonST3R_PO_Event_RGB_VGGT_pretrained_no_ev_norm.txt 2>&1


# Demo on events
# python demo_event.py --input /storage/dulanga/4DRecon/event_monst3r/data/point_odyssey/test/seminar_g110_0315_3rd --output_dir demo_tmp --seq_name seminar_g110_0315_3rd_events --weights /storage/dulanga/4DRecon/event_monst3r/results/MonST3R_PO_Event/checkpoint-final.pth --num_frames 65 --events

# Demo on RGB
# python demo_event.py --input /storage/dulanga/4DRecon/event_monst3r/data/point_odyssey/test/seminar_g110_0315_3rd --output_dir demo_tmp --seq_name seminar_g110_0315_3rd_rgb --weights /storage/dulanga/4DRecon/event_monst3r/results/MonST3R_PO_RGB/checkpoint-final.pth --num_frames 65


# evaluaton on PointOdyssey
# CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port=29604 launch.py --mode=eval_depth  \
#     --pretrained="/storage/dulanga/4DRecon/event_monst3r/results/MonST3R_PO_Event/checkpoint-best.pth"   \
#     --model="AsymmetricCroCo3DEventStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, freeze='encoder')" \
#     --eval_dataset=pointodyssey --output_dir="results/point_odyssey_event_eval" 
#     # To use the ground truth dynamic mask for davis, add: --use_gt_mask

# evaluaton on PointOdyssey
# CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port=29604 launch.py --mode=eval_depth  \
#     --pretrained="/storage/dulanga/4DRecon/event_monst3r/results/MonST3R_PO_RGB/checkpoint-best.pth"   \
#     --model="AsymmetricCroCo3DStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, freeze='encoder')" \
#     --eval_dataset=pointodyssey --output_dir="results/point_odyssey_rgb_eval" 
#     # To use the ground truth dynamic mask for davis, add: --use_gt_mask

# single depth eval on PointOdyssey EventRGB model``
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29604 launch.py --mode=eval_depth  \
#     --pretrained="/storage/dulanga/4DRecon/event_monst3r/results/MonST3R_PO_Event_RGB_VGGT_V1/checkpoint-best.pth"   \
#     --model="VGGTEventRGBStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=2048, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, freeze='encoder')" \
#     --eval_dataset=pointodyssey --output_dir="results/point_odyssey_eval" 
#     # To use the ground truth dynamic mask for davis, add: --use_gt_mask

# python depth_metric_pointodyssey.py 

# CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port=29604 launch.py --mode=eval_pose  \
#     --pretrained="/storage/dulanga/4DRecon/event_monst3r/results/MonST3R_PO_RGB/checkpoint-best.pth"   \
#     --model="AsymmetricCroCo3DEventRGBStereoV2(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, freeze='encoder')" \
#     --eval_dataset=pointodyssey --output_dir="results/point_odyssey_rgb_pose_eval" 