# JUMP.LAB — MMPose CMJ 백엔드

GitHub 저장소: https://github.com/minjae3701-cpu/CMJ-TEST

## 구성
- `main.py`: FastAPI `POST /api/analyze-pose`; `video`와 선택적 `duration_sec`를 전송하면 COCO-WholeBody 133관절 JSON 출력
- `Dockerfile`: Python 3.10 + PyTorch CPU + MMCV + MMPose + RTMPose-m WholeBody 모델 가중치
- `render.yaml`: **무료 서버만** 생성하도록 하는 배포 시도용 Blueprint (빌드 실패/메모리 부족 가능)
- `test_timing.py`: 비정상 브라우저 FPS 및 타임스탬프 보정 단위 테스트

## 테스트
`pip install -r requirements.txt pytest` 후 `pytest -q test_timing.py`.

## Render 서버
Render에서 이 저장소를 Web Service로 연결하고 Runtime=Docker, Plan=Free, Region=Singapore, Health Check=/api/health.
Render Free는 RAM 512MB이므로 MMPose 모델을 메모리에 올릴 수 없거나 빌드 실패할 가능성이 높습니다. 결제 없이 유료 플랜으로 변경하지 마세요.

서버 URL이 준비되면 Vercel `ai-jump-lab-cmj-sj` 프로젝트의 `MMPOSE_BACKEND_URL`을 `https://<render-service>.onrender.com`으로 설정하고 재배포하세요.

## 주의
- `/api/health`의 `model_configured`만으로 MMPose 실제 추론 성공을 보장할 수 없습니다.
- 현재 Vercel 중계의 업로드 상한은 약 4MB이며 영상 분석 최대 대기시간 제한이 있습니다.
- 촬영 영상은 임시 파일로 처리되며 서버에서 추론 후 삭제됩니다.
- 점프 힘 추정값은 Force Plate의 직접 측정값이 아닙니다.
