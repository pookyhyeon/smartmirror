import numpy as np
import cv2

def capture_images(cap, num_images):
    images = [None] * 4  # 4개의 위치에 대한 빈 리스트 초기화
    positions = [None] * 4  # 사용자가 지정한 위치 정보를 저장
    position_names = ['Top Left (1)', 'Top Right (2)', 'Bottom Left (3)', 'Bottom Right (4)']

    while None in images:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  # 좌우 대칭

        # 메시지 추가된 프레임을 표시용으로 사용
        display_frame = frame.copy()
        cv2.putText(display_frame, "Press 's' to capture, 'q' to quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

        # 각 위치의 선택 상태를 화면에 표시
        for i, name in enumerate(position_names):
            text = f"{name}: {'Set' if images[i] is not None else 'Empty'}"
            cv2.putText(display_frame, text, (10, 70 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                        (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('Camera', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # 's' 키를 누르면 사진 촬영
            for i in range(4):
                if images[i] is None:
                    images[i] = frame.copy()  # 메시지가 없는 실제 이미지를 저장
                    print(f"Captured for position {i+1}")
                    break
        elif key == ord('q'):  # 'q' 키를 누르면 종료
            break

        # 사진을 찍은 후 위치를 선택하게 함
        if key in [ord('1'), ord('2'), ord('3'), ord('4')]:
            position = int(chr(key)) - 1  # 선택한 위치 인덱스 (0~3)
            if images[position] is None:
                images[position] = frame.copy()  # 선택한 위치에 이미지 저장
                print(f"Photo saved in position {position+1}")

    return images

def create_collage(images):
    if None in images:
        raise ValueError("모든 위치에 사진이 배치되지 않았습니다.")
    
    # 이미지 크기 조정하기
    image_width, image_height = 640, 480
    images_resized = [cv2.resize(img, (image_width, image_height)) for img in images]

    # 흰 선 추가하기
    white_line_width = 10
    white_line_hor = np.ones((image_height, white_line_width, 3), dtype=np.uint8) * 255
    white_line_ver = np.ones((white_line_width, 2 * image_width + white_line_width, 3), dtype=np.uint8) * 255

    # 2x2 그리드로 이미지를 결합
    top_row = np.hstack([images_resized[0], white_line_hor, images_resized[1]])
    bottom_row = np.hstack([images_resized[2], white_line_hor, images_resized[3]])
    collage = np.vstack([top_row, white_line_ver, bottom_row])

    return collage

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

while True:
    print("Capture 4 photos. Press '1', '2', '3', '4' to select position and 'q' to quit.")
    images = capture_images(cap, num_images=4)

    if None not in images:
        # Create and save collage
        collage = create_collage(images)
        cv2.imwrite('photo_collage.jpg', collage)
        print("저장되었습니다.")
        cap.release()
        cv2.destroyAllWindows()
        exit()

# Release resources
cap.release()
cv2.destroyAllWindows()
