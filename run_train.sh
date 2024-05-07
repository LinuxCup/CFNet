python3 -m torch.distributed.launch \
--nproc_per_node=4 train.py \
--config config/config_mvfcev2ctx_sgd_wce_fp32_lossv2_single_newcpaug_epoch70_clip_1.py && \

python3 -m torch.distributed.launch \
--nproc_per_node=4 train.py \
--config config/config_mvfcev2ctx_sgd_wce_fp32_lossv2_single_newcpaug_epoch70_clip_2.py