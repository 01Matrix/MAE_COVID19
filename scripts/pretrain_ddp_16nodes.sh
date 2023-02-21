#!/bin/bash

### Platform check
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_GID_INDEX=3
# export NCCL_IB_HCA=mlx5_2,mlx5_5
export NCCL_IB_HCA=$(pushd /sys/class/infiniband/ > /dev/null; for i in mlx5_*; do cat $i/ports/1/gid_attrs/types/* 2>/dev/null | grep v >/dev/null && echo $i ; done; popd > /dev/null)
export NCCL_DEBUG=info
export OMP_NUM_THREADS=4
# ulimit -l 131072
export JOB_NAME=$(cat /etc/hostname | cut -d '-' -f 1,2,3)
export MASTER_FILE=$HOME/master_ip.${JOB_NAME}

export NCCL_DEBUG_SUBSYS=GRAPH,TUNING

if [ -z "$RLAUNCH_REPLICA_TOTAL" ]; then
        export RLAUNCH_REPLICA_TOTAL=1
fi

if [ -z "$RLAUNCH_REPLICA" ]; then
        export RLAUNCH_REPLICA=0
fi

if [ "$RLAUNCH_REPLICA" == "0" ]; then
        ifconfig $NCCL_SOCKET_IFNAME | grep inet | grep -v inet6 | awk '{print $2}' > ${MASTER_FILE}
fi

function finish {
        rm -rf ${MASTER_FILE}
}

trap finish EXIT INT TERM

while [ ! -f ${MASTER_FILE} ]; do
        echo "wait ${MASTER_FILE}..."
        ls > /dev/null && sleep 1;
done

export MASTER_ADDR=$(cat ${MASTER_FILE})
echo "master_ip: $MASTER_ADDR"
echo "RLAUNCH_REPLICA: $RLAUNCH_REPLICA"
echo "RLAUNCH_REPLICA_TOTAL: $RLAUNCH_REPLICA_TOTAL"

# gpu_num=`nvidia-smi | grep "V100\|A100" | wc -l`
# train_epoch_len=`expr 5000 \* ${gpu_num}`

WORLD_SIZE=16 RANK=$RLAUNCH_REPLICA MASTER_ADDR=$MASTER_ADDR MASTER_PORT=12345 python main_pretrain.py \
    --batch_size 64 \
    --local_rank 0 \
    --norm_pix_loss \
    --blr 1.5e-4 \
    --weight_decay 0.05 \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --model mae_vit_base_patch16 \
    --dataset CHE CCT C1920 CAR CCS C1000 CIC MRA MRB S_orig SIRM CXC L_orig CC_orig \
                CXSD CRDX CDX QUEX CCXD MRC CHEXD \
                CHAOSCT DL KITS LIDC LITS MMWHS \
                DR MNS MURA CXIU OCX