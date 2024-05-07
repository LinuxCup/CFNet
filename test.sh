python3 -m torch.distributed.launch --nproc_per_node=1 evaluate.py \
--config config/config_mvfcev2ctx_sgd_wce_fp32_lossv2_single_newcpaug.py \
--start_epoch 69 --end_epoch 69