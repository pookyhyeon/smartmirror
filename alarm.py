import time
import serial
#아두이노와 시리얼 통신이라 com 이거 확인 잘 하고 baud rate 도 확인  잘 해야한다. 
ser = serial.Serial(
    port='COM3',  
    baudrate=9600,
    parity=serial.PARITY_ODD,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
    timeout=0
)

count=0
def select_alarm(result):
    if result == 0:
        print(" 오래 눈감았을 경우  ")
        ser.write("c".encode())
    elif result == 1:
        print("중간 정도 ")
        ser.write("b".encode())
    else:
        print(" 처음 경고 ")
        ser.write("d".encode())

    ser.flush()  
    time.sleep(0.1)
