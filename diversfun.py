import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import jn
from numba import prange
from colorsys import hls_to_rgb
from matplotlib import gridspec
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm
from scipy.ndimage import map_coordinates
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap

def PhaseDiffuserBinning(_N, _ps=1, Seed=None):
    """
    Computes a random mask with values between -pi and pi.

    N: Size of the array side
    ps: Size of the pixels binning
    Seed: seed for the pseudorandom numbers
    """
    if Seed is None:
        Seed = np.random.randint(low=100)
    np.random.seed(Seed)
    _phasediffuser = 2*np.pi*(np.random.rand(_N//_ps, _N//_ps) - 0.5)
    _phasediffuser = np.repeat(np.repeat(_phasediffuser, _ps, axis=0), _ps, axis=1)
    return np.exp(1j*_phasediffuser)

def PhaseDiffuserGauss(_N, _cl, Seed=None):
    """
    Computes a random mask with values between -pi and pi.

    N: Size of the array side
    cl: Correlation lenght
    Seed: seed for the pseudorandom numbers
    """
    if Seed is None:
        Seed = np.random.randint(low=100)
    np.random.seed(Seed)
    noise = np.random.rand(_N, _N)
    phase = gaussian_filter(noise, sigma=_cl)
    
    phase = np.pi*phase / np.std(phase)
    return np.exp(1j*phase)

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
    
def SectionMask(size, nfigs, angle, deph=0, clock=True, Max=None, Min=None):
    """
    Generates section turning mask

    Size: Size of the array side
    nfigs: Number of masks used to map a circle
    angle: angle of the apperture
    deph: dephasing of the apperture respect to the x-axis
    clock: if True, sections turn clockwise
    Max: Cuts the values with r>Max
    Min: Cuts the values with r<Min
    """
    if Max is None:
        Max = 2
    if Min is None:
        Min = -1
    ax = np.linspace(-1, 1, size)
    axX, axY = np.meshgrid(ax, ax)

    r = np.sqrt(axX**2 + axY**2)
    theta = np.arctan2(axY, axX)

    if angle > 2*np.pi: # Considers that if the apperture angle is larger than 2pi then it was given if degrees
        dtheta = np.pi*angle/180
    if np.abs(deph) > 2*np.pi: # Considers that if the dephase angle is larger than 2pi then it was given if degrees  
        deph = np.pi*deph/180
    cthetas = np.linspace(-np.pi, np.pi, nfigs+1) + deph
    cthetas = cthetas[:-1]

    if clock:
        c = 1
    else:
        c = -1

    Masks = []
    for ctheta in cthetas:
        mask = np.zeros([size, size], dtype='complex')
        mask[r<=Max] = 1
        mask[r<=Min] = 0
        _indx, _indy = np.where((np.abs(theta-c*ctheta-2*np.pi)<=dtheta/2) | 
                                (np.abs(theta-c*ctheta+2*np.pi)<=dtheta/2) | 
                                (np.abs(theta-c*ctheta)<=dtheta/2))
        
        mask[_indx, _indy] = 0

        Masks.append(mask.astype(complex))
    return Masks

def SelectPhase(Diffuser, phase=0, dphase=0.1):
    """
    For a certain phase plate, generates a binary mask selecting the pixels with
    the same pha

    Diffuser: the phase plate
    phase: the phase with want to select
    dphase: the with of the phase window we are seleting
    """
    dephs = [-2*np.pi, 0, 2*np.pi]
    DMDMask = np.zeros(Diffuser.shape)

    for deph in dephs:
        _indx, _indy = np.where(np.abs(np.angle(Diffuser) - phase + deph)<dphase*0.5)
        DMDMask[_indx, _indy] = 1
    return DMDMask


def PhaseRampMask(N, kx, ky):
    """
    Create a phase ramp mask

    N: size of the array
    kx: wave vector in x
    ky: wave vector in y
    """
    _x = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(_x, _x)

    phase_ramp = kx*X + ky*Y
    phase = np.exp(1j*phase_ramp)
    return phase

#%% 

def HarmComb(_Imgs, _Npad=1, _axis=0, _Nper=1):
    """
    Takes the array of figures modulated in time and perform the Fourier tranforms
    to the intensity comb and respective frequencies

    Imgs: Array of data with time dependence
    Npad: Number of periods after padding
    axis: axis of 'Imgs' that represents the time dependence 
    Nper: The number of time cycles in the original signal
    """
    if _Nper != 1:
        _Imgs = np.array_split(_Imgs, _Nper, axis=_axis)[0]
        
    _img = _Imgs
    for i in range(_Npad-1):
        _Imgs = np.append(_Imgs, _img, axis=_axis)
    del _img
    
    _harms = np.fft.fftshift(np.fft.fft(_Imgs, axis=_axis), axes=_axis)
    
    _amps = np.mean(np.abs(_harms), axis=(1,2))
    del _harms
    
    _Ntot = len(_amps)
    _frqs = np.linspace(0, _Npad, _Ntot)
    _frqs = _frqs[1] - _frqs[0]
    _frqs = np.linspace(-0.5/_frqs, 0.5/_frqs, _Ntot)
    _amps = _amps/np.max(_amps)
    return _amps, _frqs

def Harmonics(_Imgs, _Nharms=4, _Npad=1, _Sym=True, _axis=0, _Nper=1):
    """
    Takes the array of figures modulated in time and perform the Fourier tranforms
    to find the harmonics

    Imgs: Array of data with time dependence (Odd number)
    Nharm: Number of Harmonics that will be extracted from the data
    Npad: Number of periods after padding
    Sym: if True extract the harmonics [-n, n]
    axis: axis of 'Imgs' that represents the time dependence 
    Nper: The number of time cycles in the original signal
    """
    if np.mod(len(_Imgs), 2) == 0:
        print('Number of images is even, please try with an odd number')
    
    if _Nper != 1:
        _Imgs = np.array_split(_Imgs, _Nper, axis=_axis)[0]
        
    if np.mod(_Npad, 2) == 0:
        _Npad += 1
        
    _img = _Imgs
    for i in range(_Npad-1):
        _Imgs = np.append(_Imgs, _img, axis=_axis)
    del _img
    
    _Ntot = len(_Imgs)
    _harms = np.fft.fftshift(np.fft.fft(_Imgs, axis=_axis), axes=_axis)
    
    _frqs = np.linspace(0, _Npad, _Ntot)
    _frqs = _frqs[1] - _frqs[0]
    _frqs = np.linspace(-0.5/_frqs, 0.5/_frqs, _Ntot)

    _inds = []
    for i in range(-_Nharms, _Nharms+1):
        _inds = np.append(_inds, np.where(np.abs(i-_frqs)<1e-6))

    _inds = _inds.astype(int)
    _harms = np.take(_harms, _inds, axis=_axis)
    
    if not _Sym:
        _harms = _harms[_Nharms:]
    return _harms

def Norm(array):
    '''
    Normalize a given 2D array to the mean of its energy

    Parameters
    ----------
    tensor : 2D complex float array
             The array to normalize

    Returns
    -------
    2D float array : A 2D array indentical to the original but
                     normalized
    '''
    return array/np.sqrt(np.mean(np.abs(array)**2))

def GetFarField(array, Npad=5, WinSize=None, Scale=None):
    '''
    Computes the far field propagation of a given array

    Parameters
    ----------
    array : 2D complex float array
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
    2D float : A 2D array given as the Fourier transform of array
    '''
    n = len(array)
    if WinSize is None:
        WinSize = n
    if Scale is None:
        Scale = 1
    pad = int((Npad-1)*n)//2
    array = np.pad(array, pad, mode='constant')
    array = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(array)))
    array = Resample(array, Scale)
    return CropCenter(array, WinSize)


def GetFarDiffuser(array, Npad=5, WinSize=None, Scale=None):
    '''
    Computes the far field propagation of a given array

    Parameters
    ----------
    array : 2D complex float array
            The array to backpropagate
    NPad : Float
           The size of the padded (N*len(array)) figure used to compute 
           the Fourier transform
    WinSize : Integer
              the size of the final figure
    Scale : Float
            Scales used to resample the image

    Returns
    -------
    2D float : A 2D array given as the inverse Fourier transform of array
    '''
    n = len(array)
    if WinSize is None:
        WinSize = n
    if Scale is None:
        Scale = 1
    pad = int((Npad-1)*n)//2
    array = np.pad(array, pad, mode='constant')
    array = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(array)))
    array = Resample(array, Scale)
    return CropCenter(array, WinSize)
    

def Resample(Array, Scale):
    '''
    Resamples a given 2D array to certain scale

    Parameters
    ----------
    Array : 2D complex float array
            The array to resample
    Scale : Integer
            the value to rescale the array

    Returns
    -------
    2D float array : A 2D array of the same size of the original but
                     stretch according to Scale
    '''
    H, W = Array.shape

    # Build coordinate grid
    y = np.linspace(-1, 1, H)
    x = np.linspace(-1, 1, W)
    Y, X = np.meshgrid(y, x, indexing='ij')

    # Scale coordinates
    Xs = X/Scale
    Ys = Y/Scale

    # Convert normalized [-1,1] → pixel coordinates [0, H-1]
    Xp = (Xs + 1) * (W - 1) / 2
    Yp = (Ys + 1) * (H - 1) / 2
    coords = np.vstack([Yp.ravel(), Xp.ravel()])

    # Interpolate real and imaginary separately
    real_part = map_coordinates(
        Array.real, coords, order=1, mode='constant', cval=0).reshape(H, W)
    imag_part = map_coordinates(
        Array.imag, coords, order=1, mode='constant', cval=0).reshape(H, W)
    return real_part + 1j * imag_part

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

def GetExperimentDMD(Diffuser, angle, nfigs, deph=0, NPad=5, clock=True, WinSize=None, Max=None, Min=None):
    """
    Generates a serie of speckle patterns as the once obtained from the experiment
    where only the intensity can be measured and using a turning section

    Diffuser: The diffuser
    angle: angle of the apperture
    nfigs: Number of masks used to map a circle
    deph: dephasing of the apperture respect to the x-axis
    Npad: the size of the padded figure used to compute the Fourier transform
    WinSize: the size of the final figure
    clock: if True, sections turn clockwise
    Max: Cuts the values with r>Max
    Min: Cuts the values with r<Min
    """
    N = Diffuser.shape[0]
    if WinSize is None:
        WinSize = N
    Masks = SectionMask(N, nfigs, angle, deph, clock, Max=Max, Min=Min)
    speckles = []
    for _mask in Masks:
        _out = GetFarField(_mask*Diffuser, NPad, WinSize)
        speckles.append(np.abs(_out)**2)
    return speckles

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

def FindVortex(_PhaseArray):
    """
    Given a phase array, finds the vortex (points where the phase is not defined)
    
    PhaseArray: the phase array 
    """
    _lenX, _lenY = _PhaseArray.shape
    _VP = []
    _VM = []
    for i in range(_lenX-1):
        for j in range(_lenY-1):
            _dp = np.unwrap([_PhaseArray[i,j], _PhaseArray[i,j+1], _PhaseArray[i+1,j+1],
                             _PhaseArray[i+1,j], _PhaseArray[i,j]])
            if np.abs(np.abs(_dp[-1]-_dp[0])-2*np.pi)<1e-2 and (i!=0 and j!=0):
                if _dp[-1]-_dp[0] > 0:
                    _VM.append([j, i])
                else:
                    _VP.append([j, i])
    return _VP, _VM

def GetGrainSize(_IntensArray, _axis=0, _p00=0.1, _p01=1e1):
    """
    Given an intensity array, computes the grain size by comparing the autocorrelation
    with the Airy spot
    
    IntensArray: the intensity array
    axis: axis to fit the airy spot
    p00: first paraleter of the fitting (the numerical apperture over/wavelenght)
    p01: second parameter of the fitting (the constant of proportionality)
    """
    _IntensArray = np.abs(Corr(_IntensArray, _IntensArray))
    _IntensArray = _IntensArray/np.max(_IntensArray)
    _len = _IntensArray.shape[_axis]
    _IntensArray = np.take(_IntensArray, indices=_len//2, axis=_axis)
    _dom = np.linspace(-_len//2, _len//2, _len)
    _opt, _cov = curve_fit(Airy, _dom, _IntensArray, p0=[_p00, _p01])
    return 1/(2*_opt[0])

def Corr(_Arr1, _Arr2):
    """
    Computes correlation of two arrays
    
    Arr1: the array 1
    Arr2: the array 2
    """
    _Arr1 = _Arr1 - np.mean(_Arr1)
    _Arr2 = _Arr2 - np.mean(_Arr2)
    _corr = np.fft.ifft2(np.conjugate(np.fft.fft2(_Arr1))*np.fft.fft2(_Arr2))
    _corr = np.fft.fftshift(_corr)
    return _corr

def FindDistances(_Pos1, _Pos2, _metric=None):
    """
    Given two set of positions, computes the distribution of minimum distances 
    between both sets, starting from the point of the set 1
    
    Pos1: the array of positions 1
    Pos2: the array of positions 2
    metric: if given, weight the elements of the distribution by this value
    """
    _dist = []
    for _v in _Pos1:
        _dist.append(np.inf)
        for _m in _Pos2:
            _d = np.sqrt((_m[0]-_v[0])**2 + (_m[1]-_v[1])**2)
            if _d < _dist[-1]:
                _dist[-1] = _d

    if _metric != None:
        _dist = _dist/_metric
    return _dist

def RadialHistogram(_dist, _bins):
    """
    Given a distribution and an axis, computes the radial probability in that
    axis
    
    dist: the distribution
    bins: the axis
    """
    _hist, _edge = np.histogram(_dist, bins=_bins, density=True)
    _edge = (_edge[:-1] + _edge[1:])/2
    _hist = _hist/_edge
    _hist = _hist/np.trapezoid(_hist*2*np.pi*_edge, _edge)
    return _hist, _edge

def Airy(_r, _a, _b):
    """
    Evaluates the Airy spot function
    
    r: Domain array
    a: the numerical apperture over/wavelenght
    b: the constant of proportionality
    """
    return (_b*jn(1, 2*np.pi*_r*_a)/_r)**2

def LeeMask(complex_pattern, period, center, angle, nbits):
    """
    Compute a binary Lee hologram mask from a complex pattern.
    Assumes |complex_pattern| <= 1.
    
    Parameters:
      complex_pattern : 2D np.complex64 array.
      period          : grating period.
      center          : tuple (center_y, center_x).
      angle           : grating angle (radians).
      nbits           : bit-depth.
      
    Returns:
      2D np.uint8 mask.
    """
    height, width = complex_pattern.shape
    omega = 2.0 * math.pi / period
    mask = np.empty((height, width), dtype=np.uint8)
    max_val = (1 << nbits) - 1  # equivalent to 2**nbits - 1
    cy, cx = center
    sin_angle = math.sin(angle)
    cos_angle = math.cos(angle)
    
    for i in prange(height):
        y = i - cy
        for j in range(width):
            x = j - cx
            tilt_angle = -sin_angle * x + cos_angle * y
            amp = abs(complex_pattern[i, j]) + math.asin(math.cos(tilt_angle * omega)) / math.pi
            amplitude_term = 1 if amp > 0.5 else 0
            phase_pattern = math.atan2(complex_pattern[i, j].imag, complex_pattern[i, j].real)
            phase_arg = (cos_angle * x + sin_angle * y) * omega - phase_pattern
            phase_term = 1 if (1.0 + math.cos(phase_arg)) / 2.0 > 0.5 else 0
            mask[i, j] = max_val if amplitude_term * phase_term else 0
    return mask
    
#%%

def useTex(_useTex):
    """
    If True activates the latex font in the plots
    """
    if _useTex:
        plt.rc('text', usetex=True)
        plt.rc('font',**{'family':'serif','serif':['Helvetica']})
    else:
        plt.rc('text', usetex=False)
        plt.rc('font', family='sans-serif')        

def Colorblind(blind):
    """
    If True activates the colorblind palette
    """
    if blind:
        sns.set_palette('colorblind')

def Colorize(array, theme="dark", saturation=1.0, beta=1.4, transparent=False,
             alpha=1.0, max_threshold=1):
    """
    Returns a vectorial array with the intensity going from black to a transparence
    that shows the phase of the array
    
    array: The complex array
    theme: if dark the background if black, if white background if white
    saturation: defines the saturation of the image
    beta: 
    transparent: if True, the plot blackground is transparent
    alpha: if transparent, defines the degree of transparency
    max_threshold: representes of the maxima of the intensity in the vectorize function 
                   which is normalized to its maxima
    """
    r = np.abs(array)
    r /= max_threshold * np.max(np.abs(r))
    arg = np.angle(array)

    h = (arg + np.pi) / (2 * np.pi) + 0.5
    l = 1.0 / (1.0 + r**beta) if theme == "white" else 1.0 - 1.0 / (1.0 + r**beta)
    s = saturation

    c = np.vectorize(hls_to_rgb)(h, l, s)  # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0, 2)
    if transparent:
        a = 1.0 - np.sum(c**2, axis=-1) / 3
        alpha_channel = a[..., None] ** alpha
        return np.concatenate([c, alpha_channel], axis=-1)
    else:
        return c
    
def ColorMaps(_CLabel=None):
    """
    Defines a colormap that goes from a black background to a bright color

    CLabel: Limite colors of the array
    """
    if _CLabel == None:
        _CLabel = 'BlackGreen'
        
    if _CLabel == 'BlackRed':
        _colors = [(0, 0, 0), (1, 0, 0)]
        _cmap = LinearSegmentedColormap.from_list('Custom', _colors, N=10000)
    elif _CLabel == 'BlackGreen':
        _colors = [(0, 0, 0), (0, 1, 0)]
        _cmap = LinearSegmentedColormap.from_list('Custom', _colors, N=10000)
    elif _CLabel == 'BlackBlue':
        _colors = [(0, 0, 0), (0, 0, 1)]
        _cmap = LinearSegmentedColormap.from_list('Custom', _colors, N=10000)
    elif _CLabel == 'RedBlue':
        _colors = [(1, 0, 0), (0, 0, 0), (0, 0, 1)]
        _cmap = LinearSegmentedColormap.from_list('Custom', _colors, N=10000)
    elif _CLabel == 'BlackPink':
        _colors = [(0, 0, 0), (159/255, 43/255, 104/255)]
        _cmap = LinearSegmentedColormap.from_list('Custom', _colors, N=10000)
    elif _CLabel == 'BlackBrown':
        _colors = [(0, 0, 0), (193/255, 154/255, 107/255)]
        _cmap = LinearSegmentedColormap.from_list('Custom', _colors, N=10000)
    elif _CLabel == '488nm':
        _colors = [(0, 0, 0), (0, 234/255, 255/255)]
        _cmap = LinearSegmentedColormap.from_list('Custom', _colors, N=10000)
    else:
        print('ColorMap not defined. Existing options: Black[Red, Green, Blue, Pink, Brown]')
    return _cmap

def Profiles(Data, save=False, Log=False, Lims=None, CMap=None, FigSize=None):
    '''
    Plots the 2D map of an array and the profile at the center of both axis
    
    Data: The array to plot
    save: if True, saves the figure
    Log: if True, the profile plots are in semi-log
    Lims: Defines the y limits of the profile plot
    CMap: Defines the colormap of the 2D array
    FigSize = Defines the size of the figure
    '''
    if FigSize is None:
        FigSize = (5,5)
    if Lims is None:
        if Log:
            Min = 1e-2
        else:
            Min = 0
        Max = 1
    if CMap is None:
        CMap = ColorMaps('BlackBlue')
    
    Spec = gridspec.GridSpec(ncols=2, nrows=2, wspace=0.05, hspace=0.05,
                             width_ratios=(1, 0.4), height_ratios=(0.4, 1))
    
    Data = Data/np.max(np.abs(Data))
    if len(Data)//2==0:
        x_data = np.mean(Data[len(Data)//2:len(Data)//2+1, :])
        y_data = np.mean(Data[:, len(Data)//2:len(Data)//2+1])
    else:
        x_data = Data[len(Data)//2, :]
        y_data = Data[:, len(Data)//2]
    
    fig = plt.figure(figsize=FigSize)
    ax = []
    ax.append(fig.add_subplot(Spec[2]))
    ax.append(fig.add_subplot(Spec[0]))
    ax.append(fig.add_subplot(Spec[3]))
    ax.append(fig.add_subplot(Spec[1]))
    
    ax[0].imshow(np.abs(Data), cmap=CMap, origin='lower')
    ax[0].axhline(len(x_data)//2+0.5, color='C1', ls='--', alpha=0.6)
    ax[0].axvline(len(x_data)//2+0.5, color='C2', ls='--', alpha=0.6)
    
    if Log:
        ax[3].imshow(np.abs(Data), cmap=CMap, origin='lower', norm=LogNorm(vmin=Min, vmax=Max))
    else:
        ax[3].imshow(np.log(np.abs(Data)), cmap=CMap, origin='lower')
    
    ax[1].plot(np.arange(len(x_data))-len(x_data)//2, np.abs(x_data), color='C1')
    ax[2].plot(np.abs(y_data), np.arange(len(y_data))-len(y_data)//2, color='C2')
    
    ax[0].axis('off')
    ax[3].axis('off')
    ax[1].set_xlim(-len(x_data)//2, len(x_data)//2)
    ax[1].set_ylim(Min, Max) 
    ax[1].xaxis.tick_top()
    ax[1].xaxis.set_label_position("top")
    ax[2].set_ylim(-len(y_data)//2, len(y_data)//2)
    ax[2].set_xlim(Min, Max)
    ax[2].yaxis.tick_right()
    ax[2].yaxis.set_label_position("right")
    if Log:
        ax[1].set_yscale('log')
        ax[1].set_ylim(Min, Max) 
        ax[2].set_xscale('log')
        ax[2].set_xlim(Min, Max)
        
    if save:
        plt.savefig('Fig.svg', bbox_inches='tight', pad_inches=0.01)
    plt.show()
