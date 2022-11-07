#!/bin/bash

### Platform check
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_2,mlx5_5
export NCCL_DEBUG=WARN  #INFO
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
echo "JOB_NAME: $JOB_NAME"

# OMP_NUM_THREADS=1 torchrun --nnodes=2 --nproc_per_node=8 --max_restarts=3 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 main_pretrain_ddp_test.py \
# OMP_NUM_THREADS=1 torchrun --nnodes=1:2 --nproc_per_node=8 --max_restarts=3 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 main_pretrain_ddp_test.py \
OMP_NUM_THREADS=1 torchrun --nnodes=2 --node_rank=$RLAUNCH_REPLICA --nproc_per_node=8 --master_addr=$MASTER_ADDR --master_port=29400 main_pretrain_ddp.py \
    --dist_url="tcp://${MASTER_ADDR}:29400" \
    --jobtype data35 \
    --batch_size 128 \
    --norm_pix_loss \
    --blr 1.5e-4 \
    --weight_decay 0.05 \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --model mae_vit_large_patch16 \
    --dataset CHE CCT C1920 CAR CCS C1000 CIC MRA MRB S_orig SIRM L_orig CC_orig \
                CXSD CRDX CDX QUEX CCXD MRC CHEXD \
                CHAOSCT DL KITS LIDC LITS MMWHS VERSE LYMPH \
                CXNIH CXPERT DR MNS MURA CXIU OCX
