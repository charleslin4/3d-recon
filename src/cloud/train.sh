#!/bin/bash

echo "Submitting AI Platform PyTorch job"

BUCKET_NAME=3d-recon
JOB_NAME=test1
JOB_DIR=gs://${BUCKET_NAME}/${JOB_NAME}/models
DATA_DIR=gs://${BUCKET_NAME}/shape_net_core_uniform_samples_2048

gcloud ai-platform jobs submit training ${JOB_NAME} \
    --region=us-west1 \
    --master-image-uri=gcr.io/cloud-ml-public/training/pytorch-gpu.1-7 \
    --scale-tier=BASIC_GPU \
    --job-dir=${JOB_DIR} \
    --package-path=./src \
    --module-name=src.train \
    # --scheduling.maxRunningTime=300s
    -- \
    --data_dir=DATA_DIR \
    --gpus=1 \

gcloud ai-platform jobs stream-logs ${JOB_NAME}

