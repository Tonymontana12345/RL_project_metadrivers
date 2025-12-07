# 주행 영상 GIF 녹화 가이드 🎬

## 📚 기능 소개

`screen_record=True`와 `env.top_down_renderer.generate_gif()`를 사용하여 주행 과정을 GIF로 녹화합니다.

**장점:**
- ✅ 시각적으로 주행 확인
- ✅ 실패 원인 파악 용이
- ✅ 프레젠테이션/리포트용
- ✅ 성공 vs 실패 비교

---

## 🚀 빠른 시작

-------
## 🚀 빠른 시작


### 1. 단일 에피소드 녹화
```bash
python Drive_Record.py --model models/sac_fixed_seed_1000.zip --seed 1000
```
**결과:** `driving.gif`

### 2. 커스텀 파일명
```bash
python Drive_Record.py --model models/sac_fixed_seed_1000.zip --seed 1000 --output my_driving.gif
```
**결과:** `my_driving.gif`

### 3. 실패하는 시드 녹화 (중요!)
```bash
python Drive_Record.py --model models/sac_fixed_seed_1000.zip --seed 2679 --output failure_seed_2679.gif
```
**결과:** 실패 지점을 GIF로 확인!

---

## 📊 다양한 사용법

### A. 여러 에피소드 녹화
```bash
python Drive_Record.py --model models/sac_fixed_seed_1000.zip --seed 2679 --episodes 3
```
**결과:**
- `gifs/seed_2679_ep1.gif`
- `gifs/seed_2679_ep2.gif`
- `gifs/seed_2679_ep3.gif`

### B. 여러 시드 비교
```bash
python Drive_Record.py --model models/sac_fixed_seed_1000.zip --seeds 1000 2679 4657 --compare
```
**결과:**
- `gifs/comparison/seed_1000.gif`
- `gifs/comparison/seed_2679.gif`
- `gifs/comparison/seed_4657.gif`

### C. 성공 vs 실패 비교 (추천! ⭐)
```bash
python Drive_Record.py --model models/sac_fixed_seed_1000.zip --success 1000 --failure 2679
```
**결과:**
- `gifs/comparison/success.gif` - 성공 사례
- `gifs/comparison/failure.gif` - 실패 사례

### D. 긴 에피소드 녹화
```bash
python Drive_Record.py --model models/sac_fixed_seed_1000.zip --seed 1000 --max-steps 2000
```
**결과:** 최대 2000 스텝까지 녹화

---

