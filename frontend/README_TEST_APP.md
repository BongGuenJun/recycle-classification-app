# 재활용품 AI 도우미 — 앱 최종 후보

Expo SDK 54 기반 카메라 앱입니다. 사용자는 `1-stage` 또는 `2-stage` 모델을 선택하고 수동·자동 촬영으로 캔·페트병·플라스틱의 재질과 오염 상태를 분석할 수 있습니다.

## 실행

```powershell
cd C:\Users\전봉근\Desktop\recycleApp\frontend
npm install
npx expo start --clear
```

앱의 API 주소는 `https://comprised-target-trapping.ngrok-free.dev`로 고정되어 있습니다. 실행 전에 백엔드와 ngrok을 각각 켜야 합니다.

```powershell
# PowerShell 1: 백엔드
cd C:\Users\전봉근\Desktop\recycleApp\backend
.\venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 30

# PowerShell 2: 외부 HTTPS 연결
ngrok http --url=comprised-target-trapping.ngrok-free.dev 8000
```

## 최종 확인

- 홈에서 1-stage·2-stage 모델 선택 및 선택값 유지
- 수동·자동 촬영과 분석
- bbox, 재질, 오염 상태, AI 확신도 표시
- 다음 물체 분석 흐름
- 최근 분석 패널과 전체 기록 페이지
- 앱 백그라운드 이동 시 카메라·센서 중지
- Wi-Fi가 아닌 모바일 데이터에서 ngrok을 통한 분석 성공

촬영 원본은 긴 변 최대 1,600px로 축소해 전송하며, 임시 파일과 최근 분석 크롭은 앱 실행 세션 단위로 관리합니다. AI 결과는 촬영 환경에 따라 달라질 수 있으며 실제 배출 기준은 지역별 안내를 함께 확인해야 합니다.
