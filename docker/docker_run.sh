# Only for running locally on t-inoue
docker run -m 10g --shm-size 10g --gpus all -it \
--volume .:/workspace/DVIS-FastInst \
--volume /home/$USER/datasets/scc_ovis_format:/workspace/DVIS-FastInst/datasets \
$USER/dvis_fastinst /bin/bash