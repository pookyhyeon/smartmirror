import time
# 지금 코드는 FPS 가 얼만큼 되는 지 측정하는 코드입니다. 현재시간과 이전시간을 빼고 그 역수를 구해서, fps 를 측정합니다 
def check_fps(prev_time) :
    cur_time = time.time()
    one_loop_time = cur_time - prev_time
    prev_time = cur_time
    fps = 1/one_loop_time
    return prev_time, fps
