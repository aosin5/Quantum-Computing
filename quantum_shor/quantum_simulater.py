import numpy as np
import math

def tensor(*operater):
    """np.kron の拡張"""
    result = operater[0]
    for i in range(1, len(operater)):
        result = np.kron(result, operater[i])
    return result


"""-------------量子状態クラスの定義---------------"""
class Qubits(object):
    def __init__(self, n_bits):
        """
        量子ビットの初期化
        Arguments: n_bits --- 量子ビット数(int)
        """
        self.n_bits    = n_bits                                     # 量子ビット数
        self.n_states  = 2**n_bits                                  # 行数
        self._amp      = np.zeros((self.n_states, 1))               # 絶対値の二乗をとると確率になる奴。超重要メンバ変数。
        self._amp[0,0] = 1                                          # 状態|00...0> の確率を1とする

    def set_bits(self, bits):
        """
        量子ビットの設定
        Arguments: bits --- 状態を表現するリスト ex).|0010> としたいなら [0,0,1,0]
        """
        idx              = int(''.join(map(str, bits)), base=2)    # bits を文字列にして結合させ、その後十進数になおす
        self._amp        = np.zeros((self.n_states, 1))            # 絶対値の二乗をとると確率になる
        self._amp[idx,0] = 1                                       # idx番目の amp を1とする
    
    def set_bit(self, k, num):
        """
        量子ビットの設定(1ビット)
        Arguments: k   --- 設定したいビット番号
                   num --- 0 or 1
        """
        # ビットの部分測定を利用する
        # k番目が 0 か 1 かを指定
        # それぞれの場合の処理を行う ( ベイズの定理的処理 )
        k_matrix  = np.array([[1,0],[0,0]])                           # 第kビットが 0 なら k_matrix = |0><0| となる
        if num == 1:                                                  # 第kビットが 1 なら
            k_matrix = np.array([[0,0],[0,1]])                        # k_matrix = |1><1| となる
        
        p_matrix  = tensor(np.eye(2**k), k_matrix, np.eye(2**(self.n_bits-k-1))) @ self._amp
        p         = self._amp.reshape((1, self.n_states)) @ p_matrix  # 0 or 1 を観測する確率
        if p == 0:
            self._amp = tensor(np.eye(2**k), np.array([[0,1],[1,0]]), np.eye(2**(self.n_bits-k-1))) @ self._amp
        else:
            self._amp = p_matrix / np.sqrt(p)    

    def measure(self):
        """
        量子ビットの観測
        """
        amp_copy  = self._amp.reshape((self.n_states))             # 下処理 self._amp から一次元配列をえる
        p         = np.abs(amp_copy)**2                            # amp_copy から観測確率を求める
        p         = p/sum(p)                                       # p の合計が1以上になったときの予防
        idx       = np.random.choice(range(len(amp_copy)), p=p)    # 適当に idx を決める。pは合計値が1となるlist
        self._amp = np.zeros((self.n_states, 1))                   # self._amp をまっさらにして
        self._amp[idx,0] = 1                                       # 一つの状態に確定させる
       
    def measure_part(self, k):
        """
        量子ビットの部分的観測
        Arguments: k --- 測定したいビットの番号(int)
        """
        # まず疑似的に全体を測定する
        # そして第kビットが 0 か 1 かを確認
        # それぞれの場合の処理を行う ( ベイズの定理的処理 )
        amp_copy  = self._amp.reshape((self.n_states))             # 下処理 self._amp から一次元配列をえる
        p         = np.abs(amp_copy)**2                            # amp_copy から観測確率を求める
        p         = p/sum(p)                                       # p の合計が1以上になったときの予防
        idx       = np.random.choice(range(len(amp_copy)), p=p)    # 適当に idx を決める。pは合計値が1となるlist
        
        b_idx     = format(idx, 'b')                               # b_idx は idx の二進数表記文字列
        k_matrix  = np.array([[1,0],[0,0]])                        # 第kビットが 0 なら k_matrix = |0><0| となる
        
        if k-(self.n_bits-len(b_idx))>=0 and b_idx[k-(self.n_bits-len(b_idx))]=="1": # 第kビットが 1 なら
            k_matrix = np.array([[0,0],[0,1]])                     # k_matrix = |1><1| となる
        
        p_matrix  = tensor(np.eye(2**k), k_matrix, np.eye(2**(self.n_bits-k-1))) @ self._amp
        p         = self._amp.reshape((1, self.n_states)) @ p_matrix
        self._amp = p_matrix / np.sqrt(p)    

    def apply(self, *operators):
        """
        量子ビットに演算子を適用
        Arguments: operators --- 行列表記の演算子(何個でも可)
        """
        for op in operators:
            self._amp = op.dot(self._amp)   # 演算子を左からかけていく
    
    def __str__(self):
        """
        print時に呼び出される表示用メソッド
        Returns: [amp]|0010>
        """
        return " + ".join(
            ("{}|{:0" + str(self.n_bits) + "b}>").format(amp, i)
            for i, amp in enumerate(self._amp) if amp
        )


"""--------------ここから下は演算子の定義---------------"""

def I(n_bits):
    return np.eye(2**n_bits)

def X(n_bits, target):
    return tensor(I(target), np.array([[0,1],[1,0]]), I(n_bits-target-1))

def H(n_bits, target):
    return tensor(I(target), np.array([[1,1],[1,-1]])/np.sqrt(2), I(n_bits-target-1))

def T(n_bits, target):
    T = np.array([[np.e**(1j*(np.pi/8)), 0], [0, np.e**(1j*(np.pi/8))]])
    return tensor(I(target), T, I(n_bits-target-1))

def SPIN(n_bits, target, spin):
    Spin = np.array([[np.e**(1j*(spin)), 0], [0, np.e**(1j*(spin))]])
    return tensor(I(target), Spin, I(n_bits-target-1))

def CNOT(n_bits, control, target):
    CNOT_zero = np.array([[1,0],[0,0]]) # 制御ビットが0の時 ( |0><0| を表している )
    CNOT_one  = np.array([[0,0],[0,1]]) # 制御ビットが1の時 ( |1><1| を表している )
    
    CNOT_zero = tensor(I(control), CNOT_zero, I(n_bits-control-1))
    CNOT_one  = tensor(I(control), CNOT_one,  I(n_bits-control-1)) @ X(n_bits, target)
    
    return CNOT_zero + CNOT_one

def CU(n_bits, control, target, U):
    m,m = U.shape
    n   = int(math.log(m,2))
    
    CU_zero = np.array([[1,0],[0,0]]) # 制御ビットが0の時 ( |0><0| を表している )
    CU_one  = np.array([[0,0],[0,1]]) # 制御ビットが1の時 ( |1><1| を表している )
    
    CU_zero = tensor(I(control), CU_zero, I(n_bits-control-1))
    CU_one  = tensor(I(control), CU_one,  I(n_bits-control-1)) @ tensor(I(target), U, I(n_bits-target-n))
    
    return CU_zero + CU_one