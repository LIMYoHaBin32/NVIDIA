##### [https://www.youtube.com/watch?v=1L8mQ6sty80&t=10s](https://www.domin.co.kr/news/photo/201605/1107650_243532_5939.jpg)

``` bash
# 1. 필요한 라이브러리 설치
!pip install ultralytics

# 2. YOLOv8 모델 사용을 위한 ultralytics 임포트
from ultralytics import YOLO
import os
from google.colab import files
import cv2
import matplotlib.pyplot as plt
import numpy as np

# 3. YOLOv8x.pt 모델 자동 다운로드 및 로드
model = YOLO('yolov8x.pt')  # YOLOv8x.pt 모델을 로드 (없으면 자동 다운로드)
print("YOLOv8x.pt 모델이 성공적으로 로드되었습니다!")

# 4. 이미지 업로드
uploaded = files.upload()  # 이미지를 업로드하는 창이 나타납니다.

# 5. 업로드된 이미지 파일 이름 확인
image_path = list(uploaded.keys())[0]  # 업로드된 첫 번째 이미지 파일 이름
print(f"업로드된 이미지: {image_path}")

# 6. YOLOv8x 모델로 객체 탐지 수행
results = model(image_path)

# 7. 결과 시각화 (라벨링된 이미지 생성)
# 원본 이미지를 로드
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR에서 RGB로 변환

# 탐지된 객체를 이미지에 그리기
for box in results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])  # 경계 상자 좌표
    class_id = int(box.cls.item())  # 텐서를 스칼라로 변환
    confidence = box.conf.item()  # 텐서를 스칼라로 변환
    label = f"{results[0].names[class_id]} {confidence:.2f}"  # 라벨과 신뢰도
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 경계 상자 그리기 (녹색)
    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # 라벨 그리기

# 8. 결과 저장
save_dir = 'runs/detect/predict'
os.makedirs(save_dir, exist_ok=True)  # 디렉토리가 없으면 생성
saved_path = os.path.join(save_dir, image_path)
cv2.imwrite(saved_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # 이미지를 저장
print(f"결과 이미지가 저장된 경로: {saved_path}")

# 9. 라벨링된 이미지 코랩에 표시
plt.figure(figsize=(12, 8))
plt.imshow(img)
plt.axis('off')  # 축 숨기기
plt.title("라벨링된 결과 이미지")
plt.show()

# 10. 저장된 파일 다운로드
if os.path.exists(saved_path):
    print(f"결과 이미지가 {saved_path}에 저장되었습니다.")
    files.download(saved_path)  # 결과 이미지를 다운로드
else:
    print("결과 이미지가 저장되지 않았습니다. 경로를 확인해주세요.")

```
![1](https://github.com/user-attachments/assets/4bf90495-495b-4e08-b039-e79879095dd5)

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

##### https://www.youtube.com/watch?v=1L8mQ6sty80&t=10s

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

# 3. YOLOv8x.pt 모델 로드
model = YOLO('yolov8x.pt')  # YOLOv8x.pt 모델을 로드 (없으면 자동 다운로드)
print("YOLOv8x.pt 모델이 성공적으로 로드되었습니다!")

# 4. 영상 업로드
uploaded = files.upload()  # 영상을 업로드하는 창이 나타납니다.
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
    # YOLOv8x로 객체 탐지
    results = model(frame_path)

    # 프레임 이미지 로드
    img = cv2.imread(frame_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR에서 RGB로 변환

    # 탐지된 객체를 이미지에 그리기
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # 경계 상자 좌표
        class_id = int(box.cls.item())  # 클래스 ID
        confidence = box.conf.item()  # 신뢰도
        class_name = results[0].names[class_id]  # 클래스 이름
        label = f"{class_name} {confidence:.2f}"  # 라벨 (예: "car 0.95", "person 0.92")
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 경계 상자 그리기 (녹색)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # 라벨 그리기

    # 라벨링된 프레임 저장
    result_path = os.path.join(result_dir, f"result_frame_{frame_idx}.jpg")
    cv2.imwrite(result_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"프레임 {frame_idx} 결과가 {result_path}에 저장되었습니다.")

    # 코랩에 라벨링된 프레임 표시
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
[results.zip](https://github.com/user-attachments/files/19596837/results.zip)
