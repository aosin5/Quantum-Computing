from quantum_simulater import *

# 指定した二つを入れ替える
def SWAP(n_bits, bit_a, bit_b):
    return CNOT(n_bits, bit_a, bit_b) @ CNOT(n_bits, bit_b, bit_a) @ CNOT(n_bits, bit_a, bit_b)

# 全体を入れ替える
def all_SWAP(n_bits):
    AS = I(n_bits)
    for i in range(int(n_bits/2)):
        AS = SWAP(n_bits, i, n_bits-1-i) @ AS
    return AS

def invRm(n_bits, control, target, m):
    U = np.array([[1, 0],
                  [0, math.e**(-2j*math.pi/2**m)]])
    return CU(n_bits, control, target, U)

def Opt_t(n_bits, target):
    RH = H(n_bits, target)
    if target == n_bits-1:
        return RH
    else:
        for m in range(2, n_bits-target+1, 1):
            RH = RH @ invRm(n_bits, target+m-1, target, m)
        return RH
    
def invQFT(n_bits):
    invQFT = all_SWAP(n_bits)
    for target in range(n_bits-1, -1, -1):
        invQFT = Opt_t(n_bits, target) @ invQFT
    return invQFT