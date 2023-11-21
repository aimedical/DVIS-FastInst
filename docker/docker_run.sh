# Only for running locally on t-inoue
docker run -m 10g --shm-size 10g --gpus all -it \
--volume .:/workspace/DVIS-FastInst \
--volume /home/$USER/datasets/ovis/ovis:/workspace/DVIS-FastInst/datasets/ovis \
$USER/dvis_fastinst /bin/bash