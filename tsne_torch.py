#
#  tsne_torch.py
#
# Implementation of t-SNE in pytorch. The implementation was tested on pytorch
# > 1.0, and it requires Numpy to read files. In order to plot the results,
# a working installation of matplotlib is required.
#
#
# The example can be run by executing: `python tsne_torch.py`
#
#
#  Created by Xiao Li on 23-03-2020.
#  Copyright (c) 2020. All rights reserved.
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import pdb
import seaborn as sns
import pickle
import os
import shutil
import pathlib
import glob
import imageio as io
from tqdm import tqdm

def str2bool(mystr):
    if mystr.lower() == "true":
        return True
    else:
        return False

def gethist(img):
    if len(img.shape) == 3:
        #img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        img = img.mean(axis=2)
        
    img = img.flatten()
    b, bins, patches = plt.hist(img, 255)

    return b 
    
    
    
def Hbeta_torch(D, beta=1.0):
    P = torch.exp(-D.clone() * beta)

    sumP = torch.sum(P)

    H = torch.log(sumP) + beta * torch.sum(D * P) / sumP
    P = P / sumP

    return H, P


def x2p_torch(X, tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape

    sum_X = torch.sum(X*X, 1)
    D = torch.add(torch.add(-2 * torch.mm(X, X.t()), sum_X).t(), sum_X)

    P = torch.zeros(n, n)
    beta = torch.ones(n, 1)
    logU = torch.log(torch.tensor([perplexity]))
    n_list = [i for i in range(n)]

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        # there may be something wrong with this setting None
        betamin = None
        betamax = None
        Di = D[i, n_list[0:i]+n_list[i+1:n]]

        (H, thisP) = Hbeta_torch(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while torch.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].clone()
                if betamax is None:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].clone()
                if betamin is None:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta_torch(Di, beta[i])

            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, n_list[0:i]+n_list[i+1:n]] = thisP

    # Return final P-matrix
    return P


def pca_torch(X, no_dims=50):
    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - torch.mean(X, 0)

    (l, M) = torch.eig(torch.mm(X.t(), X),True)
    # split M real
    for i in range(d):
        if l[i, 1] != 0:
            M[:, i+1] = M[:, i]
            i += 1

    Y = torch.mm(X, M[:, 0:no_dims])
    return Y


def tsne(X, no_dims=2, initial_dims=50, perplexity=30.0,max_iter=100):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should not have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca_torch(X, initial_dims)
    (n, d) = X.shape
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = torch.randn(n, no_dims)
    dY = torch.zeros(n, no_dims)
    iY = torch.zeros(n, no_dims)
    gains = torch.ones(n, no_dims)

    # Compute P-values
    P = x2p_torch(X, 1e-5, perplexity)
    P = torch.nan_to_num(P,0)
    P = P + P.t()
    P = P / torch.sum(P)
    P = P * 4.    # early exaggeration
    print("get P shape", P.shape)
    P = torch.max(P, torch.tensor([1e-21])) #remove zeros?

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = torch.sum(Y*Y, 1)
        num = -2. * torch.mm(Y, Y.t())
        num = 1. / (1. + torch.add(torch.add(num, sum_Y).t(), sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / torch.sum(num)
        Q = torch.max(Q, torch.tensor([1e-12]))

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = torch.sum((PQ[:, i] * num[:, i]).repeat(no_dims, 1).t() * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum

        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)).double() + (gains * 0.8) * ((dY > 0.) == (iY > 0.)).double()
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - torch.mean(Y, 0)
        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = torch.sum(P * torch.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    # Return solution
    return Y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=str)
    parser.add_argument("--outdir",type=str)
    parser.add_argument("--function",type=str,choices=["tsne","lab","mean"])
    parser.add_argument("--feattype",type=str,choices=["hist","histcolor","pix"])
    parser.add_argument("--cuda", type=int, default=1, help="if use cuda accelarate")
    parser.add_argument("--max_iter",type=int,default=100, help="max iterations of gradient descent")
    parser.add_argument("--dims",type=int,default=2,help="number of output dimensions")
    parser.add_argument("--nrimgs",type=int)

    opt = parser.parse_args()
    print("get choice from args", opt)
    
    indir = opt.indir
    outdir = opt.outdir
    #load imgs
    

    
    imgs = {}
    
    print("Loading Images")
    for ext in ["*.jpg","*.png","*.jpeg"]:
        for file in tqdm(glob.glob(os.path.join(indir,ext))[0:opt.nrimgs]):
            filename = file.split('/')[-1]
            imgs[filename] = io.imread(file)
    
    
    if opt.function == "tsne":
        
        #get histvals, later all kinds of image features can be inserted
        #should also be possible with saved features
        
        labels = list(imgs.keys())
        vals = []

        if opt.feattype == "hist":
            print("Getting Histograms")
            for label in tqdm(labels):
                vals.append(gethist(imgs[label]))
        else:
            for label in tqdm(labels):
                vals.append(imgs[label].flatten())

            
        
        if opt.cuda:
            print("set use cuda")
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        else:
            torch.set_default_tensor_type(torch.DoubleTensor)



        X = torch.Tensor(vals)

        X = X/X.max() #normalize


        with torch.no_grad():
            Y = tsne(X, opt.dims, 100, 30.0,max_iter=opt.max_iter)

        if opt.cuda:
            Y = Y.cpu().numpy()

        Y = Y.flatten()

        Y = Y - min(Y)
        Y = np.round(Y,1)
        Y = list(Y)
    
    for val,filename in zip(Y,labels):
        io.imwrite(os.path.join(outdir,str(val)+"_"+filename),imgs[filename])

        
    