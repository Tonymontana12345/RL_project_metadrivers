# 설치 가이드

MetaDrive 강화학습 프로젝트 설치 방법

---

## 📋 시스템 요구사항

### Python 버전
- **권장**: Python 3.9 - 3.11
- **비권장**: Python 3.13 (호환 문제 있음)

### 운영체제
- macOS (Apple Silicon / Intel)
- Linux
- Windows (WSL 권장)

---

## 🚀 설치 방법

### 방법 1: 자동 설치 (권장)

```bash
# 1. 프로젝트 클론 또는 다운로드
cd pgdrive_project

# 2. Python 버전 확인
python --version  # 3.9-3.11 확인

# 3. 가상환경 생성
python -m venv venv

# 4. 가상환경 활성화
# macOS/Linux:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# 5. pip 업그레이드
pip install --upgrade pip

# 6. 패키지 설치
pip install -r requirements.txt
```

### 방법 2: 단계별 설치 (문제 발생 시)

```bash
# 1. 가상환경 생성 및 활성화 (위와 동일)

# 2. 기본 의존성 먼저 설치
pip install numpy gym panda3d shapely matplotlib pillow Cython wheel

# 3. MetaDrive 설치
pip install git+https://github.com/metadriverse/metadrive.git

# 4. 강화학습 프레임워크
pip install gymnasium stable-baselines3

# 5. 시각화 및 유틸리티
pip install matplotlib seaborn tensorboard pandas tqdm pyyaml
```

---

## ✅ 설치 확인

```bash
# Python 버전 확인
python --version

# MetaDrive 설치 확인
python -c "from metadrive import MetaDriveEnv; print('MetaDrive OK')"

# 프로젝트 테스트
python quick_start.py --demo
```

**성공 메시지**:
```
MetaDrive OK
🚗 PGDrive 환경 데모
...
✅ 데모 완료!
```

---

## 🐛 문제 해결

### 문제 1: Python 3.13 사용 중

**증상**: MetaDrive 설치 실패

**해결책**:
```bash
# Python 3.9 또는 3.11 설치
# macOS (Homebrew):
brew install python@3.11

# 새 가상환경 생성
python3.11 -m venv venv_py311
source venv_py311/bin/activate

# 패키지 재설치
pip install -r requirements.txt
```

### 문제 2: MetaDrive 설치 실패

**증상**: `ModuleNotFoundError: No module named 'numpy'` 등

**해결책**:
```bash
# 의존성 먼저 설치
pip install numpy Cython wheel

# --no-build-isolation 옵션 사용
pip install --no-build-isolation git+https://github.com/metadriverse/metadrive.git
```

### 문제 3: Panda3D 설치 실패

**증상**: `error: command 'gcc' failed`

**해결책**:
```bash
# macOS: Xcode Command Line Tools 설치
xcode-select --install

# Linux: 빌드 도구 설치
sudo apt-get install build-essential python3-dev

# 재시도
pip install panda3d
```

### 문제 4: Shapely 설치 실패

**해결책**:
```bash
# macOS:
brew install geos
pip install shapely

# Linux:
sudo apt-get install libgeos-dev
pip install shapely
```

---

## 📦 패키지 버전 확인

```bash
# 설치된 패키지 확인
pip list

# 주요 패키지 버전
pip show metadrive-simulator gymnasium stable-baselines3
```

**예상 출력**:
```
Name: metadrive-simulator
Version: 0.4.x

Name: gymnasium
Version: 0.28.x

Name: stable-baselines3
Version: 2.x.x
```

---

## 🔄 업데이트

```bash
# 패키지 업데이트
pip install --upgrade -r requirements.txt

# MetaDrive 최신 버전
pip install --upgrade git+https://github.com/metadriverse/metadrive.git
```

---

## 💻 개발 환경 설정

### VS Code

```bash
# Python 확장 설치
# 1. VS Code에서 Extensions (Cmd+Shift+X)
# 2. "Python" 검색 및 설치

# 가상환경 선택
# 1. Cmd+Shift+P
# 2. "Python: Select Interpreter"
# 3. venv 선택
```

### Jupyter Notebook (선택)

```bash
# Jupyter 설치
pip install jupyter ipykernel

# 커널 등록
python -m ipykernel install --user --name=pgdrive --display-name="PGDrive"

# Jupyter 실행
jupyter notebook
```

---

## 🌐 다른 환경에서 설치

### Docker (선택)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성
RUN apt-get update && apt-get install -y \
    build-essential \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "quick_start.py"]
```

### Google Colab

```python
# Colab 노트북에서
!pip install git+https://github.com/metadriverse/metadrive.git
!pip install gymnasium stable-baselines3

# 프로젝트 파일 업로드 후 실행
!python quick_start.py --demo
```

---

## 📝 설치 체크리스트

- [ ] Python 3.9-3.11 설치 확인
- [ ] 가상환경 생성 및 활성화
- [ ] requirements.txt 설치
- [ ] MetaDrive 설치 확인
- [ ] quick_start.py 실행 성공
- [ ] (선택) GPU 설정 (PyTorch)
- [ ] (선택) Jupyter 설정

---

## 🆘 추가 도움

### 공식 문서
- [MetaDrive 문서](https://metadrive-simulator.readthedocs.io/)
- [Stable-Baselines3 문서](https://stable-baselines3.readthedocs.io/)

### 이슈 보고
- [MetaDrive GitHub Issues](https://github.com/metadriverse/metadrive/issues)

---

## 💡 팁

### 빠른 설치 확인
```bash
# 한 줄로 모든 확인
python -c "from metadrive import MetaDriveEnv; from stable_baselines3 import PPO; print('✅ All OK')"
```

### 가상환경 관리
```bash
# 가상환경 비활성화
deactivate

# 가상환경 삭제 (재설치 시)
rm -rf venv

# 새로 생성
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

**설치 완료 후 README.md를 참고하여 프로젝트를 시작하세요!**
