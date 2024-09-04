
import numpy as np
import imutils
import time
import timeit
import dlib
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
from threading import Timer
from check_cam_fps import check_fps
import make_train_data as mtd
import light_remover as lr
import ringing_alarm as alarm
import serial
import ringing_alarm 


def eye_aspect_ratio(eye) : # 눈을 통해 귀를 측정 
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def init_open_ear() : #초기화하고 7초마다 저장하는 역할 귀의 평균값을 저장함. 
    time.sleep(5)
    print("준비 중... ")
    ear_list = []
    for i in range(7) :
        ear_list.append(both_ear)
        time.sleep(1)
    global OPEN_EAR
    OPEN_EAR = sum(ear_list) / len(ear_list)

def init_close_ear() :  #임계값 설정해서 눈이 감긴지 뜬건지 구별함 
    time.sleep(2)
    th_open.join()
    time.sleep(5)
    print("준비 완료")
    ear_list = []
    th_message2 = Thread(target = init_message)
    th_message2.deamon = True
    th_message2.start()
    time.sleep(1)
    for i in range(7) :
        ear_list.append(both_ear)
        time.sleep(1)
    CLOSE_EAR = sum(ear_list) / len(ear_list)
    global EAR_THRESH
    EAR_THRESH = (((OPEN_EAR - CLOSE_EAR) / 2) + CLOSE_EAR) 

def init_message() : #시작메세지 
    print(" 졸음 감지 완료  ")
    

OPEN_EAR = 0 
EAR_THRESH = 0 
EAR_CONSEC_FRAMES = 20 # 눈감김이 지속되는 프레임 수 ( fps는 지금 camera check.py코드에서 계산함 )
COUNTER = 0  #얼만큼 지금 눈이 감기고 있는지 count 
closed_eyes_time = []
TIMER_FLAG = False 
ALARM_FLAG = False 
ALARM_COUNT = 0 
RUNNING_TIME = 0 
both_ear =0 
PREV_TERM = 0 

np.random.seed(9)
power, nomal, short = mtd.start(25) 
test_data = []
result_data = []
prev_time = 0

print("졸음운전 방지 프로그램 시작  ")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # 예측 모델 불렀음 

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("화면 시작  ")
vs = VideoStream(src=0).start()  #비디오 스트림 시작 src=0 이니까 지금 기본 카메라 사용함 
time.sleep(1.0)

th_open = Thread(target = init_open_ear)
th_open.deamon = True
th_open.start()
th_close = Thread(target = init_close_ear)
th_close.deamon = True
th_close.start()

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width = 400)
    
    L, gray = lr.light_removing(frame)
    # 일단 그레이 스케일로 프레임 읽어 온후 픽셀 조정해서 크기 축소 
    # 조명 영향을 줄이기 위해서 gray scale 진행함 
    
    rects = detector(gray,0)
# 얼굴 감지한 후 사각형 형태로 감지 
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
# 눈의 좌표 추출 후 눈이 감겼을 때 숫자를 파악 
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        both_ear = (leftEAR + rightEAR) * 500  
# 윤곽선을 그리고 녹색으로 프레임 그린다. 
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0,255,0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0,255,0), 1)
        
# 일정 수준 이상 감겼을 시 타이머 시작 , 연속일 시 counter 증가 
        if both_ear < EAR_THRESH :
            if not TIMER_FLAG:
                start_closing = timeit.default_timer()
                TIMER_FLAG = True
            COUNTER += 1

            if COUNTER >= EAR_CONSEC_FRAMES:

                mid_closing = timeit.default_timer()
                closing_time = round((mid_closing-start_closing),3)

# 정해놓은 일정 시간 초과시 경고음 울리기 시작 
                if closing_time >= RUNNING_TIME:
                    if RUNNING_TIME == 0 :
                        CUR_TERM = timeit.default_timer()
                        OPENED_EYES_TIME = round((CUR_TERM - PREV_TERM),3)
                        PREV_TERM = CUR_TERM
                        RUNNING_TIME = 1.75

                    RUNNING_TIME += 2
                    ALARM_FLAG = True
                    ALARM_COUNT += 1

                    print("{0}번째 확인 ".format(ALARM_COUNT))
                    print(" 현재 눈을 감은 시간 :", closing_time,"초")
                    test_data.append([OPENED_EYES_TIME, round(closing_time*10,3)])
                    result = mtd.run([OPENED_EYES_TIME, closing_time*10], power, nomal, short)
                    result_data.append(result)
                    t = Thread(target = alarm.select_alarm, args = (result, ))
                    t.deamon = True
                    t.start()
# 눈이 감기지 않을 경우 계속해서 초기화 
        else :
            COUNTER = 0
            TIMER_FLAG = False
            RUNNING_TIME = 0

            if ALARM_FLAG :
                end_closing = timeit.default_timer()
                closed_eyes_time.append(round((end_closing-start_closing),3))

            ALARM_FLAG = False
# 루프 종료하고 싶으면 q 누르면 종료 됩니다.
        cv2.putText(frame, "EAR : {:.2f}".format(both_ear), (300,130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,30,20), 2)

    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()

