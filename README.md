## 안드로이드 Pill Object Detection JAVA 1차 배포

### Model 설명
`asset` 폴더에 학습된 Detection Model 2가지가 있습니다.
- Stage1_OD모델 : Plate(class 1)과 Pill(Class 0)을 검출할 수 있는 초경량 모델 
- Stage2_OD모델 : Pill만 검출 가능한 중간크기의 모델 

### Stage 설명
#### Stage 1
Stage1 모델로 여러 Plate를 검출하고 그 중 약을 가장 많이 포함하고 있는 Plate를 검출하여 Stage2에 사용될 이미지 박스 위치를 구함.  
혹시 Plate를 검출하지 못하면 전체 이미지를 Stage2에서 사용

#### Stage 2
Stage2 모델로 Stage1에서 획득된 Crop된 이미지나 전체 이미지를 이용하여 한번 더 정확하게 검출

### 구현 사항 및 업데이트 예정
#### 구현사항
현재 Stage1 모델을 변환하는 과정에서 오류가 있었는지 Plate가 검출되지 않는 현상이 있습니다.  
그래서 추후 디버깅 후 Stage 1 코드를 적용할 예정입니다.  
우선 Stage2과정에 전체 이미지를 넣고 결과를 보는 코드가 적용되었습니다.  
현재 개발된 Stage2코드까지 플러터로 잘 동작한다면 그 이후는 알고리즘의 약간의 변화만 추가 될 예정이라 문제가 없을듯 합니다.
