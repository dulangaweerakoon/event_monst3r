start_time=$SECONDS
python demo_ev_voxels.py --input /storage/dulanga/4DRecon/event_monst3r/data/point_odyssey/train/human_in_scene --output_dir demo_tmp --seq_name human_in_scene_voxels --events --weights /storage/dulanga/4DRecon/event_monst3r/results2/MonST3R_PO_EventVoxels/checkpoint-best.pth --num_frames 60

# python demo_ev_voxels.py --input /storage/dulanga/4DRecon/event_monst3r/data/point_odyssey/train/human_in_scene --output_dir demo_tmp --seq_name human_in_scene_rgbflow --weights /storage/dulanga/4DRecon/event_monst3r/results2/MonST3R_PO_EventVoxels/checkpoint-best.pth --num_frames 60

duration=$((SECONDS - start_time))
printf "Time taken: %dh %dm %ds\n" $((duration/3600)) $((duration%3600/60)) $((duration%60))