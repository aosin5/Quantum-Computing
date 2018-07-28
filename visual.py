import numpy as np
import math
import matplotlib.pyplot as plt

def get_CU_gate(U):
    m,m=U.shape
    n=int(math.log(m,2))

    I=np.array([[1,0],[0,1]])
    CU_zero=np.array([[1,0],[0,0]])
    CU_one=np.array([[0,0],[0,1]])

    for i in range(n):
            CU_zero=np.kron(CU_zero,I)

    CU_one=np.kron(CU_one,U)

    return CU_zero+CU_one

def get_H_gate(U):
    m,m=U.shape
    n=int(math.log(m,2))

    H=np.array([[1,1],[1,-1]])/math.sqrt(2)
    I=np.array([[1,0],[0,1]])

    H_gate=H

    for i in range(n):
        H_gate=np.kron(H_gate,I)

    return H_gate

def hadamard_test(U,phi):

    # 初期状態を定める
    zero_bit=np.array([[1],[0]])
    state=np.kron(zero_bit,phi)

    # stateにH_gateを作用させる
    state=np.dot(get_H_gate(U),state)

    # stateにCU_gateを作用させる
    state=np.dot(get_CU_gate(U),state)

    # stateにH_gateを作用させる
    state=np.dot(get_H_gate(U),state)

    return state

def get_IV_1(state):
    m,k=state.shape
    n=int(math.log(m,2))

    PZero=sum([abs(state[i,0]**2) for i in range(2**(n-1))])

    cosL=2*PZero-1

    # 虚部が正の方を返す
    return cosL+math.sqrt(1-cosL**2)*1j

def get_IV_2(state):
    m,k=state.shape
    n=int(math.log(m,2))

    PZero=sum([abs(state[i,0]**2) for i in range(2**(n-1))])

    cosL=2*PZero-1

    # 虚部が負の方を返す
    return cosL-math.sqrt(1-cosL**2)*1j

def get_answer(U):
    w,v=np.linalg.eig(U)

    for i in range(len(w)):
        print('value:',w[i],' vector:',v[:,i])

def hadamard_loop(U,loop):
    m,m=U.shape
    n=int(math.log(m,2))
    zero=[[1],[0]]
    state=zero

    for i in range(n-1):
        state=np.kron(state,zero)

    for i in range(loop):
        end_state=hadamard_test(U,state)
        # アダマールテストで帰ってきたend_stateから１ビット目の情報を抜き取って、次のhadamard_testに使う(できれば再帰関数でかきたい)
        end_state0, end_state1=np.split(end_state,[2**n],axis=0) # end_stateを上下に分ける
        PZero=sum([abs(end_state[i,0]**2) for i in range(2**(n-1))]) # １ビット目で0が観測される確率
        if PZero==0:
            state=end_state1
        else:
            state=end_state0/((PZero)**(1/2))

    cosL=2*PZero-1
    cosL=min(cosL,1) # 1超えたときの保険
    return cosL+math.sqrt(1-cosL**2)*1j,cosL-math.sqrt(1-cosL**2)*1j
    # 返すのは固有値(ただし虚部の符号は分からないので両方返す)

def hadamard_vd(U,loop):
    data_R=[]
    data_I=[]
    num=np.arange(loop)

    for i in range(loop):
        IV_tuple=hadamard_loop(U,i+1) # 固有値を実部と虚部に分ける
        IV=IV_tuple[0] # tupleでは不便なので取り出しておく
        data_R.append(IV.real)
        data_I.append(IV.imag)

    plt.plot(num, data_R, label="IV_R")
    plt.plot(num, data_I, label="IV_I")
    plt.grid()
    plt.legend()
    plt.xlabel("num")
    plt.ylabel("IV")
    plt.show()

PI=math.pi
e=math.e

X=np.array([[0,1],[1,0]])
Y=np.array([[0,-1j],[1j,0]])
T=np.array([[e**(1j*(PI/8)),0],[0,e**(1j*(PI/8))]])


MyGate=np.kron(X,np.kron(Y,T))##input ゲート
# hadamard_vd(MyGate,10)
get_answer(X)


#--------------------------

import pygame
from pygame.locals import *
import time

def visual_vector(U,loop):
    data_R=[]
    data_I=[]
    num=np.arange(loop)

    for i in range(loop):
        IV_tuple=hadamard_loop(U,i+1) # 固有値を実部と虚部に分ける
        IV=IV_tuple[0] # tupleでは不便なので取り出しておく
        data_R.append(IV.real)
        data_I.append(IV.imag)
        # print("accuracy:",,"%")

    #answer vector
    w,v=np.linalg.eig(U)#w:固有値、v:固有ベクトル
    for i in range(len(w)):
        print('value:',w[i],' vector:',v[:,i])
        print(v[i,0])

    pygame.init()
    state=False
    Hight=600
    Width=600
    Radius=200
    Center=(int(Hight/2),int(Width/2))
    loop_count=0

    Answer=[]
    for i in range(len(w)):
        Answer.append((Center[0]+Radius*w[i].real,Center[1]+Radius*w[i].imag))
    ex_A=[]
    for j in range(len(data_I)):
        ex_A.append((Center[0]+Radius*data_R[j],Center[1]+Radius*data_I[j]))
    screen=pygame.display.set_mode((Hight,Width))
    pygame.display.set_caption("vector visualize")

    while(1):
        screen.fill((0,0,0))
        pygame.draw.circle(screen,(0,95,0),Center,Radius,2)
        angle = 180/PI*math.acos(float(w[0].real)*data_R[loop_count]+float(w[0].imag)*data_I[loop_count])
        angle_2 = 180/PI*math.pi*math.acos(w[0].real*data_R[loop_count]-w[0].imag*data_I[loop_count])
        # 双方のベクトルの内積から相対角度を求める
        print("loop:", loop_count+1, "  angle", min(angle,angle_2), "°")
        if state == True:
            loop_count -= 1
        for answer_num in range(len(Answer)):
            pygame.draw.line(screen,(0,0,100),Center,Answer[answer_num],3)
        pygame.draw.line(screen,(0,100,100),Center,ex_A[loop_count],3)
        loop_count+=1
        time.sleep(1)
        pygame.display.update()
        for event in pygame.event.get():
            #終了用イベント
            if event.type==QUIT:
                pygame.quit()
            #キー入力時
            if event.type==pygame.KEYDOWN:
                if event.key==K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key==pygame.K_SPACE: state=not(state)

MyGate2=X@X@T@Y@T
MyGate1=X@T
MyGate0=X

visual_vector(MyGate2,30)
