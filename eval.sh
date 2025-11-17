# single depth eval on PointOdyssey EventRGB model``
CUDA_VISIBLE_DEVICES=8 torchrun --nproc_per_node=1 --master_port=29604 launch.py --mode=eval_depth  \
    --pretrained="/storage/dulanga/4DRecon/event_monst3r/results2/MonST3R_PO_Event_RGB_VGGT_pretrained/checkpoint-best.pth"   \
    --model="VGGTEventRGBStereo(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=2048, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, freeze='encoder')" \
    --eval_dataset=pointodyssey --output_dir="results2/point_odyssey_eval" 
    # To use the ground truth dynamic mask for davis, add: --use_gt_mask

python depth_metric_pointodyssey.py 
