# NMFをPythonで実装してみる  

[前回](https://leck-tech.com/machine-learning/nmf)、NMF(非負値行列因子分解)について理論面でのまとめをした。  
今回はその内容を踏まえて実装を行ってみる。  

使うモジュールは以下の通り。  

```
python : 3.7.4 

numpy
matplotlib
```

## 1.NMFの実装  

### 1.1 Divergenceごとの更新式の実装  

#### 1.1.1 Squared Euclidean Distance 

二乗ユークリッド距離でのパラメータ更新式は以下の通り。  

$$
B_{mk} =  B_{mk} \frac{(XW^T)_{mk}}{(BWW^T)_{mk}} 
$$

$$ 
W_{kn} = W_{kn} \frac{(B^TX)_{kn}}{(B^TBW)_{kn}}　
$$ 

基本的に、足し算引き算で実装するため、簡略化して、  

$$
params = \eta ((分子) - (分母)) 
$$
で実装した。  

```python 
def update_matrix(X, B, W, eta=1e-3): 
    w_diff = eta * (np.dot(B.T, X) - np.dot(np.dot(B.T, B), W))
    b_diff = eta * (np.dot(X, W.T) - np.dot(np.dot(B,W), W.T)) 
    return b_diff, w_diff
```

#### 1.1.2 Kullback-Leibler Divergence

今度はKullback-Leibler Divergence。  

$$
B_{mk} = B_{mk}\frac{\sum_n W_{kn}(X_{mn}/(BW)_{mn})}{\sum_n W_{kn}} 
$$

$$
W_{kn} = W_{kn} \frac{\sum_m B_{mk}(X_{mn}/(BW)_{mn})}{\sum_m B_{mk}} 
$$

ここでは式が成り立つように $\eta$ の前に $(BまたはW)/(分母)$ をかけることで正規化をはかった。  

```python
 def update_matrix(X, B, W, eta=1e-3): 
    _X = np.dot(B, W) 
    w_diff_denom = B.T.sum(axis=1, keepdims=True) 
    b_diff_denom = W.T.sum(axis=0, keepdims=True)
    w_diff =  eta * W/w_diff_denom * (np.dot(B.T, X/_X) -w_diff_denom) 
    b_diff = eta * B/b_diff_denom * (np.dot(X/_X, W.T) - b_diff_denom)
    return b_diff, w_diff  
```

#### 1.1.3 板倉・斎藤 Divergence 

板倉・斎藤(IS) Divergence の更新式とコードは以下のようになる。  

$$
B_{mk} = B_{mk} \frac{\sum_n W_{kn}(X_{mn}/(BW)_{mn}^2)}{\sum_nW_{kn}(1/(BW)_{mn})}
$$

$$
W_{kn} = W_{kn}\frac{\sum_mB_{mk}(X_{mn}/(BW)^2_{mn})}{\sum_m B_{mk}(1/(BW)_{mn})} 
$$

```python 
def update_matrix(X, B, W, eta=1e-3):
    _X_ = X/np.dot(B,W)**2 
    w_diff_denom = np.dot(B.T, 1.0/np.dot(B,W))
    b_diff_denom = np.dot(1.0/np.dot(B,W), W.T)
    w_diff = eta * W/w_diff_denom * (np.dot(B.T, _X_) - w_diff_denom)
    b_diff = eta * B/b_diff_denom * (np.dot(_X_, W.T) - b_diff_denom) 
    return b_diff, w_diff
```

### 1.2 学習部分の実装  

とりあえず初期値を0~1の一様乱数で初期化する。  

```python
eta = 1e-3 
num_iter = 1000
# X が入力。(M, N). 
B = np.random.rand(M, K)
W = np.random.rand(K, N)
for i in range(num_iter): 
    b_diff, w_diff = update_matrix(X, B, W, eta=eta) 
    if b_specified: 
        pass 
    else: 
        B += b_diff 
    W += w_diff  
```

### 1.3 実行コード  

適当な画像を作って、それに対してNMFをかけてみた。  


```python 
row = 100 
col = 200
X = np.sin(np.arange(row)*0.02*np.pi).reshape(row, 1) * np.cos(np.arange(col)*0.04*np.pi).reshape(1, col) + 1.0
plt.figure(figsize=(8,5)) 
plt.title("Original X") 
plt.imshow(X) 
plt.savefig("./org_X.png")
plt.close() 
```
これを表示させてみると、以下のような画像が表示される。 
![org_X](https://leck-tech.com/wp-content/uploads/2019/12/org_X.png)  

では、これらを先ほど作った3つの指標を元に作った更新式でNMFを実行する。  

イテレーションの回数(`num_iter`)と学習率(`eta`)の2つの値を調整した。  
`NMF(.)` 関数部分は他の実装も含めて記事の最後に掲載する。  

```python 
for mode in ["EU", "KL", "IS"]: 
    if mode == "EU": 
        num_iter = 1000 
        eta = 1e-3
    elif mode == "KL": 
        num_iter = 10000
        eta = 1e-2 
    elif mode == "IS": 
        num_iter = 10000
        eta = 1e-2
    B, W = NMF(X, K=5, divergence=mode, 
                num_iter=num_iter, 
                eta=eta) 
    plt.figure(figsize=(10,8)) 
    plt.subplot(2,1,1)
    plt.title("Original X")
    plt.imshow(X) 
    plt.subplot(2,1,2) 
    plt.title(f"Reconstructed with {mode}")
    plt.imshow(np.dot(B, W)) 
    plt.savefig(f"./{mode}_NMF.png") 
    plt.close() 
```

再構築された画像はそれぞれ以下のようになる。  

- Euclidean Distance 

![EU_NMF.png](https://leck-tech.com/wp-content/uploads/2019/12/EU_NMF.png)  

- Kullback-Leibrer Divergence 
![KL_NMF.png](https://leck-tech.com/wp-content/uploads/2019/12/KL_NMF.png)  
- 板倉・斎藤 Divergence 
![IS_NMF.png](https://leck-tech.com/wp-content/uploads/2019/12/IS_NMF-1.png)  

これくらい単純な画像であればどの評価指標でもまあまあうまくいくらしい。    

## まとめ  

今回はNMFについてPythonで実装をしてみた。  
色々な解析で使えるようにコードを整えておけると良いなと思った。  
今度は音源分離をやってみたい。  

## Appendix 

最後に、今回実装したコードの全体を掲載しておく。  
$B$が指定されれば $W$のみを学習させるようにした。  
```python 
import numpy as np 
import matplotlib.pyplot as plt 

def NMF(X, B=None, K=3, divergence="sparse-KL", num_iter=100, eta=1e-3): 
    """
    Execute Nonnegative Matrix Factorization (NMF). 
    Args :  
        X : (M, N). ndarray. input. Each element should be positive. 
        B : (M, K). ndarray. (optional). If specified, 
                    only Weight matrix will be updated. 
        K : int. Decomposition dimension. 
        divergence : {"sparse-KL", "EU", "KL", "IS"}. Metrics for divergence. 
            "sparse-KL" : Kullback Leibler Divergence with sparse norm. 
            "EU" : Squared Euclidean Divergence. 
            "KL" : Kullback Leibler Divergence. 
            "IS" : Itakura Saito Divergence. 
        num_iter : int. Number of iterations.
        eta : float. Updating rate. 
    Outputs : 
        B : (M, K). ndarray. Basis matrix. 
        W : (K, N). ndarray. Weight matrix. 
    """
    if X.min() < 0: 
        raise ValueError("X should be positive")
    M = X.shape[0] 
    N = X.shape[1] 
    
    # Define update function. 
    if divergence == "EU": 
        def update_matrix(X, B, W, eta=1e-3): 
            w_diff = eta * (np.dot(B.T, X) - np.dot(np.dot(B.T, B), W))
            b_diff = eta * (np.dot(X, W.T) - np.dot(np.dot(B,W), W.T)) 
            return b_diff, w_diff
    elif divergence == "KL": 
        def update_matrix(X, B, W, eta=1e-3): 
            _X = np.dot(B, W) 
            w_diff_denom = B.T.sum(axis=1, keepdims=True) 
            b_diff_denom = W.T.sum(axis=0, keepdims=True)
            w_diff =  eta * W/w_diff_denom * (np.dot(B.T, X/_X) -w_diff_denom) 
            b_diff = eta * B/b_diff_denom * (np.dot(X/_X, W.T) - b_diff_denom)
            return b_diff, w_diff 
    elif divergence == "IS": 
        def update_matrix(X, B, W, eta=1e-3):
            _X_ = X/np.dot(B,W)**2 
            w_diff_denom = np.dot(B.T, 1.0/np.dot(B,W))
            b_diff_denom = np.dot(1.0/np.dot(B,W), W.T)
            w_diff = eta * W/w_diff_denom * (np.dot(B.T, _X_) - w_diff_denom)
            b_diff = eta * B/b_diff_denom * (np.dot(_X_, W.T) - b_diff_denom) 
            return b_diff, w_diff 
    elif divergence == "sparse-KL": 
        def update_matrix(X, B, W, eta=1e-3): 
            raise NotImplementedError 
    if type(B) != np.ndarray: 
        B = np.random.rand(M, K)
        b_specified = False
    else: 
        b_specified = True 
    B = np.random.rand(M, K)
    W = np.random.rand(K, N)
    X_max = X.max() 
    X /= X_max 
    for i in range(num_iter): 
        b_diff, w_diff = update_matrix(X, B, W, eta=eta) 
        if b_specified: 
            pass 
        else: 
            B += b_diff 
        W += w_diff 
    W *= X_max
    return B, W

if __name__ == "__main__": 
    row = 100 
    col = 200
    X = np.sin(np.arange(row)*0.02*np.pi).reshape(row, 1) * np.cos(np.arange(col)*0.04*np.pi).reshape(1, col) + 1.0
    plt.figure(figsize=(5,8)) 
    plt.title("Original X") 
    plt.imshow(X) 
    plt.savefig("./org_X.png")
    plt.close() 
    for mode in ["EU", "KL", "IS"]: 
        if mode == "EU": 
            num_iter = 1000 
            eta = 1e-3
        elif mode == "KL": 
            num_iter = 10000
            eta = 1e-2 
        elif mode == "IS": 
            num_iter = 10000
            eta = 1e-2
        B, W = NMF(X, K=5, divergence=mode, 
                    num_iter=num_iter, 
                    eta=eta) 
        plt.figure(figsize=(10,8)) 
        plt.subplot(2,1,1)
        plt.title("Original X")
        plt.imshow(X) 
        plt.subplot(2,1,2) 
        plt.title(f"BW with {mode}")
        plt.imshow(np.dot(B, W)) 
        plt.savefig(f"./{mode}_NMF.png") 
        plt.close() 
        
```