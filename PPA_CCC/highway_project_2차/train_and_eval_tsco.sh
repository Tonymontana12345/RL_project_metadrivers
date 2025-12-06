#!/bin/bash
# TSCO 맵 학습 및 평가 자동화 스크립트

cd /Users/rainis/Desktop/school/2학기_강화학습기초/프로젝트/hiway/highway_project

echo "============================================================"
echo "🚗 TSCO 맵 학습 및 평가 자동화"
echo "============================================================"
echo ""

# 학습 시작
echo "📚 1단계: 학습 시작..."
./venv_pgdrive/bin/python train.py --mode fixed --algorithm ppo

# 학습 완료 확인
if [ ! -f "models/ppo_tsco_map_500k.zip" ]; then
    echo "❌ 학습 실패 또는 모델 파일을 찾을 수 없습니다."
    exit 1
fi

echo ""
echo "============================================================"
echo "✅ 학습 완료!"
echo "============================================================"
echo ""

# 평가 실행
echo "📊 2단계: 평가 시작..."
./venv_pgdrive/bin/python evaluate.py --model models/ppo_tsco_map_500k.zip --episodes 20 --save tsco_evaluation_results.json

echo ""
echo "============================================================"
echo "✅ 학습 및 평가 완료!"
echo "============================================================"
echo ""
echo "📁 결과 파일:"
echo "  - 모델: models/ppo_tsco_map_500k.zip"
echo "  - 평가 결과: results/tsco_evaluation_results.json"
echo ""

