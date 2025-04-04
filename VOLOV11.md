##### https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSuu5rdKS-7wGMd2oq_KV1FVx_2VabwpCrJ8g&s

``` bash
from ultralytics import YOLO

# 모델 로드
model = YOLO("yolo11n.pt")

# 업로드한 파일 이름으로 변경
results = model("images.jpeg")  # 실제 업로드한 파일 이름 사용
results[0].show()  # 결과 표시
```
////////////////////////////////////////////////////////////////////////////////////////
![image](https://github.com/user-attachments/assets/25068f05-1cb3-4d1a-aeb1-384e5e106870)

##### https://www.youtube.com/watch?v=fMot3yjoaFQ

``` bash
# 1. 필요한 라이브러리 설치
!pip install ultralytics

# 2. 필요한 라이브러리 임포트
from ultralytics import YOLO
import cv2
import os
from google.colab import files
import matplotlib.pyplot as plt
import numpy as np

# 3. YOLOv11n.pt 모델 로드 (Nano 버전, 필요 시 다른 버전으로 변경)
model = YOLO('yolo11n.pt')  # YOLOv11n.pt 모델 로드 (없으면 자동 다운로드)
print("YOLOv11n.pt 모델이 성공적으로 로드되었습니다!")

# 4. 영상 업로드
uploaded = files.upload()  # 영상을 업로드하는 창이 나타남
video_path = list(uploaded.keys())[0]  # 업로드된 영상 파일 이름
print(f"업로드된 영상: {video_path}")

# 5. 영상에서 프레임 추출 (10 프레임 단위)
cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 총 프레임 수
fps = int(cap.get(cv2.CAP_PROP_FPS))  # 초당 프레임 수
print(f"총 프레임 수: {frame_count}, FPS: {fps}")

# 프레임 저장 디렉토리 생성
frame_dir = 'frames'
os.makedirs(frame_dir, exist_ok=True)

# 10 프레임 단위로 추출
frame_list = []
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx % 10 == 0:  # 10 프레임 단위로 추출
        frame_path = os.path.join(frame_dir, f"frame_{frame_idx}.jpg")
        cv2.imwrite(frame_path, frame)  # 프레임 저장
        frame_list.append((frame_idx, frame_path))
    frame_idx += 1
cap.release()
print(f"추출된 프레임 수: {len(frame_list)}")

# 6. 각 프레임에 대해 객체 탐지 및 라벨링
result_dir = 'results'
os.makedirs(result_dir, exist_ok=True)  # 결과 저장 디렉토리 생성

for frame_idx, frame_path in frame_list:
    # YOLOv11로 객체 탐지
    results = model(frame_path)

    # 프레임 이미지 로드
    img = cv2.imread(frame_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR에서 RGB로 변환

    # 탐지된 객체를 이미지에 그리기 (이미지 스타일 반영)
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # 경계 상자 좌표
        class_id = int(box.cls.item())  # 클래스 ID
        confidence = box.conf.item()  # 신뢰도
        class_name = results[0].names[class_id]  # 클래스 이름
        label = f"{class_name} {confidence:.2f}"  # 라벨 (예: "person 0.88")

        # 파란색 경계 상자 그리기 (이미지 스타일 반영)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 파란색 (BGR: 255, 0, 0)

        # 라벨 배경 상자 (검정색 반투명 배경)
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        top_left = (x1, y1 - label_height - 10 if y1 - label_height - 10 > 0 else y1 + 20)
        bottom_right = (x1 + label_width, top_left[1] + label_height + baseline)
        cv2.rectangle(img, top_left, bottom_right, (0, 0, 0), cv2.FILLED)  # 검정색 배경

        # 라벨 텍스트 (흰색)
        cv2.putText(img, label, (x1, y1 - 10 if y1 - 10 > 0 else y1 + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # 흰색 텍스트

    # 라벨링된 프레임 저장
    result_path = os.path.join(result_dir, f"result_frame_{frame_idx}.jpg")
    cv2.imwrite(result_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"프레임 {frame_idx} 결과가 {result_path}에 저장되었습니다.")

    # Colab에 라벨링된 프레임 표시
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"프레임 {frame_idx} - 라벨링된 결과")
    plt.show()

# 7. 결과 이미지를 ZIP 파일로 압축해서 다운로드
!zip -r results.zip results  # 결과 폴더를 ZIP으로 압축
files.download('results.zip')  # ZIP 파일 다운로드
print("모든 라벨링된 프레임이 results.zip으로 다운로드되었습니다.")
```
![Uploading image.png…]()




