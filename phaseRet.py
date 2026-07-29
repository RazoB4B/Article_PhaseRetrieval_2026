#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 11:17:51 2026

@author: alberto-razo
"""
import time
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm


def VortexMask(size, ordmax=1, Sym=True, Max=None, Min=None):
    '''
    Generates spiral phase mask exp(i n phi)

    Parameters
    ----------
    Size : Integer
           Size of the array side
    ordmax : Integer
             Maximum order of the mask
    Sym : Boolean
          If True, generate mask considering [-n, n]
    Max : Float in [0, 2]
          Cuts the values with r>Max
    Min : Float in [0, 2]
          Cuts the values with r<Min

    Returns
    -------
    2D float : A stack of 2D arrays with an amplitude step function 
               defined by Min and Max, and with phases exp(i n phi)
    '''
    if Max is None:
        Max = 2
    if Min is None:
        Min = -1
    ax = np.linspace(-1, 1, size)
    axX, axY = np.meshgrid(ax, ax)

    r = np.sqrt(axX**2 + axY**2)
    theta = np.arctan2(axY, axX)
    
    if Sym:
        inds = np.arange(-ordmax, ordmax+1)
    else:
        inds = np.arange(0, ordmax+1)
    
    Masks = []
    for i in inds:
        phase = (i*theta+np.pi)%(2*np.pi)-np.pi

        mask = np.zeros([size, size], dtype='complex')
        indX, indY = np.where((r<=Max) & (r>=Min))
        mask[indX, indY] = np.exp(-1j*phase[indX, indY])

        Masks.append(mask.astype(complex))
    if ordmax == 0:
        return Masks[0]
    else:
        return Masks


def GetFarField_Torch(array, Npad=5, WinSize=None, Scale=None):
    '''
    Computes the far field propagation of a given tensor

    Parameters
    ----------
    array : 2D complex float tensor
            The array to propagate
    NPad : Float
           The size of the padded (N*len(array)) figure used to compute
           the Fourier transform
    WinSize : Integer
              the size of the final figure
    Scale : Float
            Scales used to resample the image

    Returns
    -------
    2D float tensor : A 2D tensor given as the Fourier transform of tensor
    '''
    n = array.shape[-1]
    if WinSize is None:
        WinSize = n

    if Scale is None:
        Scale = 1
    
    pad = int((Npad - 1) * n)// 2
    # Pad (left, right, top, bottom) — note reversed order in F.pad for 2D
    array_padded = F.pad(array, (pad, pad, pad, pad), mode='constant', value=0)
    # Apply fftshift, fft2, then fftshift again
    shifted_input = torch.fft.fftshift(array_padded, dim=(-2, -1))
    fft_output = torch.fft.fft2(shifted_input)
    shifted_fft = torch.fft.fftshift(fft_output, dim=(-2, -1))
    shifted_fft = Resample_Torch(shifted_fft, Scale)
    return CropCenter_Torch(shifted_fft, WinSize)


def CropCenter(array, crop):
    '''
    Crops the central section of a given array

    Parameters
    ----------
    array : 2D complex float array
            The array to crop
    crop : Integer
           Number of pixels size we want extract

    Returns
    -------
    2D float array : A 2D array given as the center of the original
                     one with size crop*crop
    '''
    x, y = array.shape
    startx = x//2-(crop//2)
    starty = y//2-(crop//2)
    return array[startx:startx+crop, starty:starty+crop]


def CropCenter_Torch(tensor, size):
    '''
    Crops the central section of a given tensor

    Parameters
    ----------
    tensor : 2D complex float tensor
             The tensor to crop
    crop : Integer
           Number of pixels size we want extract

    Returns
    -------
    2D float tensor : A 2D tensor given as the center of the original
                      one with size crop*crop
    '''
    center = tensor.shape[-1] // 2
    half = size // 2

    tensor = tensor[..., center - half:center - half + size, center - half:center - half + size]
    return tensor


def Resample_Torch(tensor, scale):
    '''
    Resamples a given 2D tensor to certain scale

    Parameters
    ----------
    tensor : 2D complex float tensor
             The tensor to resample
    Scale : Integer
            the value to rescale the tensor

    Returns
    -------
    2D float tensor : A 2D tensor of the same size of the original but
                      stretch according to Scale
    '''
    device = tensor.device
    H, W = tensor.shape
    y = torch.linspace(-1, 1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)
    Y, X = torch.meshgrid(y, x, indexing='ij')

    grid = torch.stack((X, Y), dim=-1)
    scaled_grid = grid/scale

    U_real = tensor.real.unsqueeze(0).unsqueeze(0)
    U_imag = tensor.imag.unsqueeze(0).unsqueeze(0)

    sampled_real = torch.nn.functional.grid_sample(
        U_real, scaled_grid.unsqueeze(0), mode='bilinear', align_corners=True)

    sampled_imag = torch.nn.functional.grid_sample(
        U_imag, scaled_grid.unsqueeze(0), mode='bilinear', align_corners=True)
    return sampled_real.squeeze() + 1j * sampled_imag.squeeze()


def Norm(tensor):
    '''
    Normalize a given 2D tensor to the mean of its amplitude

    Parameters
    ----------
    tensor : 2D complex float tensor
             The tensor to normalize

    Returns
    -------
    2D float tensor : A 2D tensor indentical to the original but
                      normalized
    '''
    return tensor/torch.mean(torch.abs(tensor))


def L2_Norm_Torch(x, y):
    '''
    Computes the norm in 2D between two different tensors

    Parameters
    ----------
    x : 2D complex float tensor
        The tensor 1
    y : 2D complex float tensor
        The tensor 2

    Returns
    -------
    Float : The L2 difference between two tensors
    '''
    return torch.sqrt(torch.mean(torch.abs(x - y)**2))


def GetFourierConstants(alpha, deph=0):
    '''
    Computes the coeffitiens of the fourier serie used in the algorithm

    Parameters
    ----------
    alpha : Float
            The angle of the apperture of the temporal mask
    deph : Float
           A dephasing factor that add an acoustic phase to the coeffitiens

    Returns
    -------
    2 Floats : The complex coeffitients of the Fourier serie
    '''
    Nangle = (2*np.pi)/(alpha)
    cm1 = np.abs(np.sinc(1/Nangle))/(Nangle-1)*np.exp(-1j*deph)
    cp1 = np.abs(np.sinc(1/Nangle))/(Nangle-1)*np.exp(1j*deph)
    return cp1, cm1


class EarlyStopping:
    '''
    Early stopping function for the torch optimizer

    Initialization : EarlyS = EarlyStopping(Patience=100, Mindelta=0)
    Usage : EarlyS(Error)

    Parameters
    ----------
    Patience : Integer
               Number of steps after the code stops
    Mindelta : Float
               Minimum value of decreasing of the error to consider the
               optimizer is not improving
    Loss : Float
           Error to minimice by the optimizer
    '''
    def __init__(self, Patience=500, Mindelta=0):
        self.Patience = Patience
        self.Mindelta = Mindelta
        self.Counter = 0
        self.BestLoss = None
        self.should_stop = False

    def __call__(self, Loss):
        if self.BestLoss is None or Loss < self.BestLoss-self.Mindelta:
            self.BestLoss = Loss
            self.Counter = 0
        else:
            self.Counter += 1
            if self.Counter >= self.Patience:
                self.should_stop = True


class IncreasingThreshold:
    '''
    Function for the torch optimizer. Since it is considered a certain maximum of
    error to accept or not a diffuser, and that such maximum can be hardly achieve
    when considering systems with disorder.

    Initialization : Threshold = IncreasingThreshol(Patience=5, Increase=0.05)
    Usage : MaxLoss = Threshold(MaxLoss, Loss)
    
    Parameters
    ----------
    Patience : Integer
               Number of times the solver will restar before increasing the threshold
    Increase : Float
               Increase of the maximum threshold
    MaxLoss : Float
              Current threshold
    Loss : Float
           Error to minimice by the optimizer 
    '''
    def __init__(self, Patience=5, Increase=0.05):
        self.Patience = Patience
        self.Increase = Increase
        self.Counter = 0

    def __call__(self, MaxLoss, Loss):
        self.Counter += 1
        if self.Counter >= self.Patience:
            self.Counter = 0
            print(f'Increasing maximum loss to {self.Increase + MaxLoss}, Loss={Loss}')
            return self.Increase + MaxLoss
        else:
            print(f'Restarting {self.Counter}/{self.Patience-1}, Loss={Loss}')
            return MaxLoss


class SavingBest:
    '''
    Function for the torch optimizer. Keeps the best diffuser ever found if it is not
    good enough for the current threshold. If threshold becomes larger than the storaged
    diffuser, then this array is automatically taken.

    Initialization : Best = SavingBest()
    Usage : Best(Diff, Loss, Times, Scale)
    
    Parameters
    ----------
    Diff : 2D real float tensor
           The phase values of the diffuser
    Loss : Float array
           Error to minimice by the optimizer
    Times : Float array
            The time scale used to solve the system
    Scale : Float tensor
            The scale obtained for Diff
    '''
    def __init__(self):
        self.Diff = None
        self.Loss = None
        self.Scale = None
        self.Times = None

    def __call__(self, Diff, Loss, Times, Scale):
        if ((self.Loss is None) or (Loss[-1]<self.Loss[-1])):
            self.Loss = Loss
            self.Diff = Diff
            self.Scale = Scale
            self.Times = Times
            print(f'Saving best diffuser with loss {self.Loss[-1]} and scale {self.Scale.item()}')


def Start(MaxSteps, pbar=None):
    '''
    Function for the torch optimizer. When the gradient descent loop starts or restarts
    it initializes all the varibles of the optimizations.

    Parameters
    ----------
    MaxSteps : int
               The maximum number of steps
    pbar : Boolean
           Restarts the process bar

    Returns
    -------
    The 1D empty arrays for the Losses, and the Elapsedtime
    '''
    if pbar is not None:
        pbar.close()
    TotLoss = []
    ElapTime = []
    timei = time.time()
    _step = 0
    pbar = tqdm(total=MaxSteps)

    return TotLoss, ElapTime, timei, _step, pbar


def FindDiffuser(Input, LR_init=1, LR_prop=0.01, deph=0, TryR=False, InitSca=None, InitDiff=None, 
        DiffSize=None, MaxSteps=None, MaxLoss=None, NPad=None):
    '''
    Optimizer based of grandient descent to find the near field pattern that generates
    certain Far field Harmonics (Used for small array <100x100 if not InitDiff).
    
    Parameters
    ----------
    Input : Experimental to minimize (a_0, H_0 and H_1)
    LR_init : Initial learning rate
    LR_prop : Proportion of the initial learning rate that is apply to the scale variable
    deph : dephasing of the spatiotemporal modulated pattern with respect to the x-axis (if known)
    TryR : Determines if the loops should restart is a solution is found and it does not
           match with the MaxLoss allowed
    InitSca : An initial guess of the PSF size with respect to the one generated by
              a full pupil
    InitDiff : An initial guess of the diffuser to find
    DiffSize : The size of the diffuser to find (if it is different than the size of the
               input images)
    MaxSteps : Maximun number of steps before giving up
    MaxLoss : Maximum loss tolerate (~0.3 for perfect systems and ~0.5 for experiments)
    NPad : Padding value of the optical support
    
    Returns
    -------
    Diff : The found diffuser
    Scale : The found scale
    TotLoss : The loss as function of the epoch
    ElapTime : The time as function of the epoch
    '''
    device = "cpu"
    EarlyS = EarlyStopping(Patience=100, Mindelta=0)
    Threshold = IncreasingThreshold(Patience=5, Increase=0.05)
    Best = SavingBest()

    # Recieving and computing numpy arrays
    cp1, cm1 = GetFourierConstants(np.pi, deph)
    
    a02 = Input[0]
    S0 = Input[1]
    S1 = Input[2]
    SizeArray = a02.shape[0]

    if MaxLoss is None:
        MaxLoss = 0.3

    if MaxSteps is None:
        MaxSteps = int(1e4)
        
    if DiffSize is None:
        DiffSize = SizeArray

    if InitSca is None:
        InitSca = 1

    if NPad is None:
        NPad = 10

    VMasks = VortexMask(DiffSize, 1, Max=1)

    #Loading to torch
    a02_target = Norm(torch.from_numpy(a02).to(torch.complex64).to(device))
    S0_target = Norm(torch.from_numpy(S0).to(torch.complex64).to(device))
    S1_target = Norm(torch.from_numpy(S1).to(torch.complex64).to(device))

    V0 = torch.from_numpy(VMasks[1]).to(torch.complex64).to(device)
    Vp1 = torch.from_numpy(VMasks[2]).to(torch.complex64).to(device)
    Vm1 = torch.from_numpy(VMasks[0]).to(torch.complex64).to(device)
    
    scale = torch.nn.Parameter(torch.log(torch.tensor([InitSca], device=device)))
    if InitDiff is None:
        param_diff = 2*np.pi*(torch.rand(DiffSize, DiffSize, requires_grad=True) - 0.5)
        param_diff = param_diff.clone().detach().requires_grad_(True)
    else:
        param_diff = torch.tensor(np.angle(InitDiff), dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([{'params':param_diff, 'lr':LR_init},
                                  {'params':scale, 'lr':LR_init*LR_prop}])

    # Optimization
    TotLoss, ElapTime, timei, _step, pbar = Start(MaxSteps, pbar=None)
    while _step < MaxSteps:
        optimizer.zero_grad()
    
        diff = torch.exp(1j*param_diff)
        scale_eff = torch.exp(scale)

        pred_a0 = Norm(GetFarField_Torch(diff*V0, NPad, WinSize=SizeArray, Scale=scale_eff))
        pred_ap1 = cp1*Norm(GetFarField_Torch(diff*Vp1, NPad, WinSize=SizeArray, Scale=scale_eff))
        pred_am1 = cm1*Norm(GetFarField_Torch(diff*Vm1, NPad, WinSize=SizeArray, Scale=scale_eff))

        a02_pred = Norm(torch.abs(pred_a0)**2)
        S0_pred = Norm(torch.abs(pred_a0)**2 + torch.abs(pred_am1)**2 + torch.abs(pred_ap1)**2)
        S1_pred = Norm(pred_a0*pred_am1.conj() + pred_ap1*pred_a0.conj())

        loss_a0 = L2_Norm_Torch(a02_pred, a02_target)
        loss_S0 = L2_Norm_Torch(S0_pred, S0_target)
        loss_S1 = L2_Norm_Torch(S1_pred, S1_target)

        loss_total = loss_a0 + loss_S0 + loss_S1

        loss_total.backward()
        optimizer.step()

        TotLoss.append(loss_total.item())
        ElapTime.append(time.time() - timei)
        _step += 1
        pbar.update(1)

        EarlyS(loss_total.item())
        if TryR and (EarlyS.should_stop and loss_total.item()>MaxLoss):
            MaxLoss = Threshold(MaxLoss, loss_total.item())
            Best(param_diff, TotLoss[:_step], ElapTime[:_step], scale_eff)

            if Best.Loss[-1] < MaxLoss:
                print(f'Best diffuser with loss {Best.Loss[-1]} has been previously found')
                param_diff = Best.Diff
                scale_eff = Best.Scale
                TotLoss = Best.Loss
                ElapTime = Best.Times
            else:
                EarlyS = EarlyStopping(Patience=100, Mindelta=0)
                param_diff = torch.nn.Parameter(2*np.pi*(torch.rand(DiffSize, DiffSize, device=device) - 0.5))
                scale = torch.nn.Parameter(torch.log(torch.tensor([InitSca], device=device)))

                optimizer = torch.optim.Adam([{'params': param_diff, 'lr': LR_init},
                                              {'params': scale, 'lr': LR_init * LR_prop}])
                TotLoss, ElapTime, timei, _step, pbar = Start(MaxSteps, pbar=pbar)
        if EarlyS.should_stop:
            print(f'Found - Loss={TotLoss[-1]} - Scale={scale_eff.item()}')
            Scale = scale_eff.item()
            TotLoss = TotLoss[:_step]
            ElapTime = ElapTime[:_step]
            break

    Diff = np.exp(1j*param_diff.detach().numpy())
    return Diff, Scale, TotLoss, ElapTime


def FindBigDiffuser(Input, Div=2, LR_init=1, LR_prop=0.01, deph=0, InitSca=None, DiffSize=None, 
                    MaxSteps=None, MaxLoss=None, NPad=None):
    '''
    Optimizer based of grandient descent to find the near field pattern that generates
    certain Far field Harmonics.
    This function uses FindDiffuser at every step.
    
    Parameters
    ----------
    Input : Experimental to minimize (a_0, H_0 and H_1)
    Div : Number of times the search is divided
    LR_init : Initial learning rate
    LR_prop : Proportion of the initial learning rate that is apply to the scale variable
    deph : dephasing of the spatiotemporal modulated pattern with respect to the x-axis (if known)
    InitSca : An initial guess of the PSF size with respect to the one generated by
              a full pupil
    DiffSize : The size of the diffuser to find (if it is different than the size of the
               input images)
    MaxSteps : Maximun number of steps before giving up
    MaxLoss : Maximum loss tolerate (~0.3 for perfect systems and ~0.5 for experiments)
    NPad : Padding value of the optical support
    
    Returns
    -------
    Diff : The found diffuser
    Scale : The found scale
    TotLoss : The loss as function of the epoch
    ElapTime : The time as function of the epoch
    '''
    a02= Input[0]
    S0 = Input[1]
    S1 = Input[2]
    SizeArray = a02.shape[0]
    
    if DiffSize is None:
        DiffSize = SizeArray

    for i in np.flip(range(Div+1)):
        _a02 = CropCenter(a02, SizeArray//(2**i))
        _S0 = CropCenter(S0, SizeArray//(2**i))
        _S1 = CropCenter(S1, SizeArray//(2**i))
        
        _DiffSize = DiffSize//(2**i)

        if i == Div:
            print(f'Starting with a size of {len(_a02)}x{len(_a02)} pixels')
            Diff, Scale, _TotLoss, _ElapTime = FindDiffuser([_a02, _S0, _S1], LR_init, LR_prop, deph, True, 
                    InitSca=InitSca, DiffSize=_DiffSize, MaxSteps=MaxSteps, MaxLoss=MaxLoss, NPad=NPad)
        else:
            print(f'Increasing the size of the system to {len(_a02)}x{len(_a02)} pixels')
            Diff, Scale, _TotLoss, _ElapTime = FindDiffuser([_a02, _S0, _S1], LR_init/2, LR_prop, deph, False, 
                    InitSca=Scale, InitDiff=Diff, DiffSize=_DiffSize, MaxSteps=MaxSteps, MaxLoss=MaxLoss, NPad=NPad)

        if i != 0:
            Diff = np.repeat(np.repeat(Diff, 2, axis=0), 2, axis=1)

        try:
            TotLoss = np.append(TotLoss, _TotLoss)
            ElapTime = np.append(ElapTime, _ElapTime+ElapTime[-1])
        except:
            TotLoss = _TotLoss
            ElapTime = _ElapTime

    return Diff, Scale, TotLoss, ElapTime
