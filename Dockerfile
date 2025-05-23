# 베이스 이미지: Python 3.10 (slim)
FROM python:3.10-slim

# 필수 패키지 설치 (ffmpeg 등 필요시 여기 추가)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리
WORKDIR /app

# requirements.txt 복사 및 설치
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

# 포트 설정 (FastAPI 기본 포트는 8000, 예시에서는 8010)
EXPOSE 8376

# 실행 명령
CMD ["uvicorn", "action:app", "--host", "0.0.0.0", "--port", "8376"]
