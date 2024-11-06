import whisper
import torch

# GPU 사용 가능 여부 확인
print("CUDA 사용 가능:", torch.cuda.is_available())

# 모델 로드 테스트
model = whisper.load_model("base")
print("모델 로드 성공!")