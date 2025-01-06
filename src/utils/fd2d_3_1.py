"""2D FDTD, TM program"""

from typing import Tuple
from math import exp, sin, pi

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d





def fdtd_3D_data(display: bool, dims: list[int]) -> Tuple[np.ndarray, np.ndarray] | None:
    # Grid parameters
    xAxisSize: int = dims[0]
    yAxisSize: int = dims[1]
    xAxisCenter = xAxisSize // 2
    yAxisCenter = yAxisSize // 2

    # Step parameters
    ddx = 0.01
    dt = ddx / 6e8

    # Dielectric profile
    epsilon_z = 8.854e-12

    # Pulse parameters
    t0 = 20
    spread = 6

    TEMPORAL_DIMENSION = 100


    ez, dz, hx, hy, gaz, ihx, ihy = initFields([xAxisSize, yAxisSize])
    plotting_points = initResutsToSave([40, 100])

    gaussianSource = {
        'type': 'gaussian',
        't0': t0,
        'spread': spread,
        'position': [xAxisCenter, yAxisCenter]
    }
    sineSource = {
        'type': 'sine',
        'frequency': 1500 * 1e6,
        'timeCellSize': dt,
        'position': [xAxisCenter - 5, yAxisCenter - 5]
    }

    gi2, gi3, fi1, fi2, fi3, gj2, gj3, fj1, fj2, fj3 = initPMLParams(xAxisSize)
    paramArraysDict = {
        'gi2': gi2, 'gi3': gi3, 'fi1': fi1, 'fi2': fi2, 'fi3': fi3, 
        'gj2': gj2, 'gj3': gj3, 'fj1': fj1, 'fj2': fj2, 'fj3': fj3
    }
    createPML(8, paramArraysDict, xAxisSize, yAxisSize)

    fields: dict[str, np.ndarray] = {'ez': ez, 'dz': dz, 'hx': hx, 'hy': hy, 'gaz': gaz, 'ihx': ihx, 'ihy': ihy}
    mainFDTDLoop(
        fields,
        TEMPORAL_DIMENSION,
        [xAxisSize, yAxisSize],
        sineSource,
        plotting_points,
        paramArraysDict
    )

    if not display:
        return getDataToVTK(1, plotting_points)
    else:
        plotSavedResults(
            [xAxisSize, yAxisSize],
            plotting_points,
            1,
            2
        )
        return None



def mainFDTDLoop(
        fields: dict[str, np.ndarray], 
        temporalDim: int,
        spatialDims: list[int],
        sourceProfile: dict,
        plotting_points: list[dict],
        parametersPML: dict[str, np.ndarray]
        ) -> None:
    
    ez, dz, hx, hy, gaz, ihx, ihy = fields['ez'], fields['dz'], fields['hx'], fields['hy'], fields['gaz'], fields['ihx'], fields['ihy']
    gi2, gi3, fi1, fi2, fi3 = parametersPML['gi2'], parametersPML['gi3'], parametersPML['fi1'], parametersPML['fi2'], parametersPML['fi3']
    gj2, gj3, fj1, fj2, fj3 = parametersPML['gj2'], parametersPML['gj3'], parametersPML['fj1'], parametersPML['fj2'], parametersPML['fj3']

    for time_step in range(1, temporalDim + 1):
        # Calculate Dz
        for j in range(1, spatialDims[1]):
            for i in range(1, spatialDims[0]):
                dz[i, j] = gi3[i] * gj3[j] * dz[i, j] + \
                gi2[i] * gj2[j] * 0.5 * \
                (hy[i, j] - hy[i - 1, j] - hx[i, j] + hx[i, j - 1])
        
        if sourceProfile['type'] == 'gaussian':
            pulse = setGaussianSource(
                sourceProfile['t0'],
                time_step,
                sourceProfile['spread']
            )
        elif sourceProfile['type'] == 'sine':
            pulse = setSineSource(
                sourceProfile['frequency'],
                sourceProfile['timeCellSize'],
                time_step
            )
        dz[sourceProfile['position'][0], sourceProfile['position'][1]] = pulse

        # Calculate Ez from Dz
        for j in range(1, spatialDims[1]):
            for i in range(1, spatialDims[0]):
                ez[i, j] = gaz[i, j] * dz[i, j]
        
        # Calculate the Hx field
        for j in range(spatialDims[1] - 1):
            for i in range(spatialDims[0] - 1):
                curl_e = ez[i, j] - ez[i, j + 1]
                ihx[i, j] = ihx[i, j] + curl_e
                hx[i, j] = fj3[j] * hx[i, j] + fj2[j] * \
                (0.5 * curl_e + fi1[i] * ihx[i, j])

        # Calculate the Hy field
        for j in range(spatialDims[1] - 1):
            for i in range(spatialDims[0] - 1):
                curl_e = ez[i, j] - ez[i + 1, j]
                ihy[i, j] = ihy[i, j] + curl_e
                hy[i, j] = fi3[i] * hy[i, j] - fi2[i] * \
                (0.5 * curl_e + fj1[j] * ihy[i, j])
        
        # Save the data at certain points for later plotting
        for plotting_point in plotting_points:
            if time_step == plotting_point['num_steps']:
                plotting_point['data_to_plot'] = np.copy(ez)
                plotting_point['data_to_save'] = [np.copy(hx), np.copy(hy)]
               



def initFields(dims: list[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ez = np.zeros((dims[0], dims[1]))
    dz = np.zeros((dims[0], dims[1]))
    hx = np.zeros((dims[0], dims[1]))
    hy = np.zeros((dims[0], dims[1]))
    gaz = np.ones((dims[0], dims[1]))
    ihx = np.zeros((dims[0], dims[1]))
    ihy = np.zeros((dims[0], dims[1]))

    return ez, dz, hx, hy, gaz, ihx, ihy


def initResutsToSave(num_steps: list[int]) -> list[dict]:
    labels = ['a', 'b', 'c', 'd']
    plotting_points = []
    for label, num_step in zip(labels, num_steps):
        plotting_points.append(
            {'label': label, 'num_steps': num_step, 'data_to_plot': None, 'data_to_save': None}
        )
    return plotting_points


def setGaussianSource(t0: int, time_step: int, spread: float) -> float:
    return exp(-0.5 * ((t0 - time_step) / spread) ** 2)

def setSineSource(freqency: float, timeCellSize: float, time_step: int) -> float:
    return sin(2 * pi * freqency * timeCellSize * time_step)

def plotSavedResults(dims: list[int], plotting_points: list[dict], nrow: int, ncol: int) -> None:
    plt.style.use(plt.style.available[6])
    plt.rcParams['font.size'] = 12
    plt.rcParams['grid.color'] = 'midnightblue'
    plt.rcParams['grid.linestyle'] = 'dotted'
    fig = plt.figure(figsize=(8, 7))

    X, Y = np.meshgrid(range(dims[0]), range(dims[1]))

    for subplot_num, plotting_point in enumerate(plotting_points):
        ax = fig.add_subplot(nrow, ncol, subplot_num + 1, projection='3d')
        plot_e_field(ax,
                    plotting_point['data_to_plot'],
                    X,
                    Y,
                    plotting_point['num_steps'],
                    plotting_point['label'],
                    dims
                    )
    
    plt.subplots_adjust(bottom=0.05, left=0.10, hspace=0.05)
    plt.show()


def plot_e_field(ax, data: np.ndarray, X: np.ndarray, Y: np.ndarray, timestep: int, label: str, dims: list[int]):
    """3D plot of E field at a single time step"""
    ax.set_zlim(0, 1)
    ax.view_init(elev=20, azim=45)
    ax.plot_surface(X, Y, data[:, :], rstride=1, cstride=1, color="darkgreen",
                   edgecolor='midnightblue', lw=0.25)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r' $E_Z$', rotation=90, labelpad=10, fontsize=14)
    ax.set_zticks([0, 0.5, 1])
    ax.set_xlabel('cm')
    ax.set_ylabel('cm')
    ax.set_xticks(np.arange(0, dims[0] + 1, step=20))
    ax.set_yticks(np.arange(0, dims[1] + 1, step=20))
    ax.text2D(0.6, 0.7, "T = {}".format(timestep), transform=ax.transAxes)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    plt.gca().patch.set_facecolor('k')
    ax.text2D(-0.2, 0.8, "({})".format(label), transform=ax.transAxes)
    ax.dist = 11


def getDataToVTK(dataNum: int, savedData: list[dict]) -> Tuple[np.ndarray, np.ndarray]:
    if dataNum in range(len(savedData)):
        return savedData[dataNum]['data_to_save'][0], savedData[dataNum]['data_to_save'][1]
    else:
        print('Index out of range...')
        return savedData[0]['data_to_save'][0], savedData[0]['data_to_save'][1]


def initPMLParams(xdimension: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                            np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gi2 = np.ones(xdimension)
    gi3 = np.ones(xdimension)
    fi1 = np.zeros(xdimension)
    fi2 = np.ones(xdimension)
    fi3 = np.ones(xdimension)

    gj2 = np.ones(xdimension)
    gj3 = np.ones(xdimension)
    fj1 = np.zeros(xdimension)
    fj2 = np.ones(xdimension)
    fj3 = np.ones(xdimension)

    return gi2, gi3, fi1, fi2, fi3, gj2, gj3, fj1, fj2, fj3


def createPML(npml: int, paramArraysDict: dict[str, np.ndarray], xdim: int, ydim: int) -> None:
    for n in range(npml):
        xnum = npml  - n
        xd = npml
        xxn = xnum / xd
        xn = 0.33 * xxn ** 3

        paramArraysDict['gi2'][n] = 1 / (1 + xn)
        paramArraysDict['gi2'][xdim - 1 - n] = 1 / (1 + xn)
        paramArraysDict['gi3'][n] = (1 - xn) / (1 + xn)
        paramArraysDict['gi3'][xdim - 1 - n] = (1 - xn) / (1 + xn)

        paramArraysDict['gj2'][n] = 1 / (1 + xn)
        paramArraysDict['gj2'][ydim - 1 - n] = 1 / (1 + xn)
        paramArraysDict['gj3'][n] = (1 - xn) / (1 + xn)
        paramArraysDict['gj3'][ydim - 1 - n] = (1 - xn) / (1 + xn)

        xxn = (xnum - 0.5) / xd
        xn = 0.33 * xxn ** 3

        paramArraysDict['fi1'][n] = xn
        paramArraysDict['fi1'][xdim - 2 - n] = xn
        paramArraysDict['fi2'][n] = 1 / (1 + xn)
        paramArraysDict['fi2'][xdim - 2 - n] = 1 / (1 + xn)
        paramArraysDict['fi3'][n] = (1 - xn) / (1 + xn)
        paramArraysDict['fi3'][xdim - 2 - n] = (1 - xn) / (1 + xn)

        paramArraysDict['fj1'][n] = xn
        paramArraysDict['fj1'][ydim - 2 - n] = xn
        paramArraysDict['fj2'][n] = 1 / (1 + xn)
        paramArraysDict['fj2'][ydim - 2 - n] = 1 / (1 + xn)
        paramArraysDict['fj3'][n] = (1 - xn) / (1 + xn)
        paramArraysDict['fj3'][ydim - 2 - n] = (1 - xn) / (1 + xn)



if __name__ == "__main__":
    display = True
    dims = [60, 60]
    result = fdtd_3D_data(display, dims)

    if result is not None:
        hx, hy = result[0], result[1]
        print(hx)

