#!/bin/bash


# 실행할 프로세스 수를 정의합니다.
num_folds=5  # Top Attn Images

source .venv/bin/activate

# 프로세스를 시작합니다.
for ((fold_num=0; fold_num<3; fold_num++)); do
    python step2_TAB_SCARF.py $fold_num&  # 백그라운드에서 Python 스크립트 실행
    sleep 5
done
wait

for ((fold_num=3; fold_num<5; fold_num++)); do
    python step2_TAB_SCARF.py $fold_num&  # 백그라운드에서 Python 스크립트 실행
    sleep 5
done

wait


echo "모든 프로세스가 종료되었습니다."
