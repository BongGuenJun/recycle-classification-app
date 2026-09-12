# 재현용 노트북

Google Colab에서 번호 순서대로 실행하는 것을 기준으로 정리했습니다. 코드 파일명은 실행 환경의 호환성을 위해 영어로 유지하고, 설명은 한국어로 작성했습니다.

1. `01_prepare_raw_dataset.ipynb`: AI Hub 원본 파일 확인 및 재현 가능한 train/validation 분할
2. `02_build_yolo_datasets.ipynb`: 1-stage 9클래스·2-stage 3클래스 YOLO 데이터 생성
3. `03_train_one_stage_yolo.ipynb`: 1-stage YOLOv8n 학습
4. `04_train_two_stage_detector.ipynb`: 2-stage 재질 탐지기 학습
5. `05_create_classifier_crops.ipynb`: GT bbox 기반 오염도 분류 crop 생성
6. `06_train_resnet18_classifier.ipynb`: padding 후보 비교 및 ResNet18 학습
7. `07_validate_end_to_end.ipynb`: IoU matching, threshold sweep, macro F1 기반 내부 검증
8. `08_evaluate_external_dataset.ipynb`: 실사용 데이터, 조건별 성능, Oracle crop 평가

## 원본 데이터

- AI Hub: [재활용품 분류 및 선별 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71362)
- AI Hub 이용 승인이 필요한 원본 이미지와 어노테이션은 저장소에 포함하지 않습니다.
- 각 노트북 상단의 `TEAM_PROJECT` 또는 데이터 경로만 자신의 Google Drive 구조에 맞게 확인합니다.

## 공개 범위

노트북은 데이터 변환, 모델 학습, 내부 End-to-End 검증, 외부 평가 흐름을 확인할 수 있도록 제공합니다. GPU 종류, 라이브러리 버전, 무작위 연산에 따라 재실행 결과는 README의 기록값과 소폭 달라질 수 있습니다.
