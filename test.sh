python3 -m torch.distributed.launch --master_port 65535 --nproc_per_node=1 evaluate.py \
--config config/config_mvfcev2ctx_sgd_wce_fp32_lossv2_single_newcpaug_epoch70_clip_5_nu.py \
--start_epoch 69 --end_epoch 69