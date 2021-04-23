#!/bin/bash

echo "Submitting AI Platform PyTorch job"

BUCKET_NAME=3d-recon
JOB_NAME=test
JOB_DIR=gs://${BUCKET_NAME}/${JOB_NAME}/models

gcloud ai-platform jobs submit training ${JOB_NAME} \
    --region=us-west1 \
    --scale-tier=BASIC_GPU \\
    --job-dir=${JOB_DIR} \
    --package-path=./src \
    --module-name=src.train \
    --scheduling.maxRunningTime=300s

