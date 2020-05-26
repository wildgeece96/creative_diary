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
        

     



