#!/bin/bash
# SubImageLocator 启动脚本

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export PATH=/home/ubuntu/miniconda3/envs/image_matching/bin:/usr/bin:/bin
export PYTHONUNBUFFERED=1

cd /home/ubuntu/Disk/codes/jianxiong/SubImageLocator
python app.py
