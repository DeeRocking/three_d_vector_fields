"""2D FDTD, TM program"""

from typing import Tuple, List
from math import exp, sin, pi

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d
import numba





def fdtd_3D_data(display: bool, dims: list[int], targetIndex: int, animationData: bool) -> dict | None:
    # Grid parameters
    xAxisSize: int = dims[0]
    yAxisSize: int = dims[1]
    zAxisSize: int = dims[2]
    xAxisCenter = xAxisSize // 2
    yAxisCenter = yAxisSize // 2
    zAxisCenter = zAxisSize // 2

    # Step parameters
    ddx = 0.01
    dt = ddx / 6e8

    # Dielectric profile
    epsilon_z = 8.854e-12

    # Pulse parameters
    t0 = 20
    spread = 6

    TEMPORAL_DIMENSION = 60


    fields = initFields([xAxisSize, yAxisSize, zAxisSize])
    specifyDipole(fields['gaz'], [xAxisCenter, yAxisCenter, zAxisCenter])

    if animationData:
        plotting_points = initResutsToSave([i for i in range(TEMPORAL_DIMENSION)], animationData)
    else:
        plotting_points = initResutsToSave([30, 40, 50, 60], False)

    gaussianSource = {
        'type': 'gaussian',
        't0': t0,
        'spread': spread,
        'position': [xAxisCenter, yAxisCenter, zAxisCenter]
    }
    sineSource = {
        'type': 'sine',
        'frequency': 1500 * 1e6,
        'timeCellSize': dt,
        'position': [xAxisCenter - 5, yAxisCenter - 5, zAxisCenter - 5]
    }


    mainFDTDLoop(
        fields,
        TEMPORAL_DIMENSION,
        [xAxisSize, yAxisSize, zAxisSize],
        sineSource,
        plotting_points,
        animationData
    )

    if not display:
        if not animationData:
            return getDataToVTK(
                targetIndex, 
                plotting_points, 
                [xAxisCenter, yAxisCenter, zAxisCenter], 
                False)
        else:
            return getDataToVTK(
                targetIndex, 
                plotting_points, 
                [xAxisCenter, yAxisCenter, zAxisCenter],True)
    else:
        nrow, ncol = 2, 2
        plotSavedResults(
            [xAxisSize, yAxisSize],
            plotting_points,
            nrow,
            ncol,
            zAxisCenter
        )
        return None



def mainFDTDLoop(
        fields: dict[str, np.ndarray], 
        temporalDim: int,
        spatialDims: list[int],
        sourceProfile: dict,
        plotting_points: list[dict],
        animationData: bool
        ) -> None:
    
    ex, ey, ez = fields['ex'], fields['ey'], fields['ez']
    hx, hy, hz = fields['hx'], fields['hy'], fields['hz']
    dx, dy, dz = fields['dx'], fields['dy'], fields['dz']
    gax, gay, gaz = fields['gax'], fields['gay'], fields['gaz']

    for time_step in range(1, temporalDim + 1):
        # Calculate Dz
        dx, dy, dz = calculate_d_fields(
            spatialDims,
            (dx, dy, dz),
            (hx, hy, hz)
        )

        # Add the source
        
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
        dz[sourceProfile['position'][0], sourceProfile['position'][1], sourceProfile['position'][1]] = pulse

        # Calculate the E field from the D field
        ex, ey, ez = calculate_e_fields(
            spatialDims,
            (dx, dy, dz),
            (gax, gay, gaz),
            (ex, ey, ez)
        )
        
        # Calculate the H field
        hx, hy, hz = calculate_h_fields(
            spatialDims,
            (hx, hy, hz),
            (ex, ey, ez)
        )
        
        # Save the data at certain points for later plotting
        if not animationData:
            for plotting_point in plotting_points:
                if time_step == plotting_point['num_steps']:
                    plotting_point['data_to_plot'] = np.copy(ez)
                    plotting_point['data_to_save'] = [np.copy(ex), np.copy(ey), np.copy(ez)]
        else:
            for plotting_point in plotting_points:
                plotting_point['data_to_save'] = [np.copy(ex), np.copy(ey), np.copy(ez)]


@numba.jit(nopython=True)
def calculate_d_fields(
        dims: list[int], 
        dField: Tuple[np.ndarray, np.ndarray, np.ndarray],
        hField: Tuple[np.ndarray, np.ndarray, np.ndarray]
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the Dx, Dy and Dz fields"""
    ie, je, ke = dims[0], dims[1], dims[2]
    dx, dy, dz = dField[0], dField[1], dField[2]
    hx, hy, hz = hField[0], hField[1], hField[2]

    for i in range(1, ie):
        for j in range(1, je):
            for k in range(1, ke):
                dx[i, j, k] = dx[i, j, k] + 0.5 * \
                (hz[i, j, k] - hz[i, j - 1, k] - \
                 hy[i, j, k] + hy[i, j, k - 1])

    for i in range(1, ie):
        for j in range(1, je):
            for k in range(1, ke):
                dy[i, j, k] = dy[i, j, k] + 0.5 * \
                (hx[i, j, k] - hx[i, j, k - 1] - \
                 hz[i, j, k] + hz[i - 1, j, k])
    
    for i in range(1, ie):
        for j in range(1, je):
            for k in range(1, ke):
                dz[i, j, k] = dz[i, j, k] + 0.5 * \
                (hy[i, j, k] - hy[i - 1, j, k] - \
                 hx[i, j, k] + hx[i, j - 1, k])

    return dx, dy, dz


@numba.jit(nopython=True)
def calculate_e_fields(
        dims: list[int],
        dField: Tuple[np.ndarray, np.ndarray, np.ndarray],
        gaField: Tuple[np.ndarray, np.ndarray, np.ndarray],
        eField: Tuple[np.ndarray, np.ndarray, np.ndarray]
        ) ->Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the E field from the D field"""

    ie, je, ke = dims[0], dims[1], dims[2]
    dx, dy, dz = dField[0], dField[1], dField[2]
    gax, gay, gaz = gaField[0], gaField[1], gaField[2]
    ex, ey, ez = eField[0], eField[1], eField[2]

    for i in range(0, ie):
        for j in range(0, je):
            for k in range(0, ke):
                ex[i, j, k] = gax[i, j, k] * dx[i, j, k]
                ey[i, j, k] = gay[i, j, k] * dy[i, j, k]
                ez[i, j, k] = gaz[i, j, k] * dz[i, j, k]

    return ex, ey, ez


@numba.jit(nopython=True)
def calculate_h_fields(
        dims: list[int], 
        hField: Tuple[np.ndarray, np.ndarray, np.ndarray],
        eField: Tuple[np.ndarray, np.ndarray, np.ndarray]
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the Dx, Dy and Dz fields"""

    ie, je, ke = dims[0], dims[1], dims[2]
    ex, ey, ez = eField[0], eField[1], eField[2]
    hx, hy, hz = hField[0], hField[1], hField[2]

    for i in range(0, ie):
        for j in range(0, je - 1):
            for k in range(0, ke - 1):
                hx[i, j, k] = hx[i, j, k] + 0.5 * \
                (ey[i, j, k+1] - ey[i, j, k] - \
                 ez[i, j + 1, k] + ez[i, j, k])
    
    for i in range(ie - 1):
        for j in range(0, je):
            for k in range(0, ke - 1):
                hy[i, j, k] = hy[i, j, k] + 0.5 * \
                (ez[i + 1, j, k] - ez[i, j, k] - \
                 ex[i, j, k + 1] + ex[i, j, k])
    
    for i in range(0, ie - 1):
        for j in range(0, je - 1):
            for k in range(0, ke):
                hz[i, j, k] = hz[i, j, k] + 0.5 * \
                (ex[i, j + 1, k] - ex[i, j, k] - \
                 ey[i + 1, j, k] + ey[i, j, k])

    return hx, hy, hz



def initFields(dims: list[int]) -> dict[str, np.ndarray]:
    ex = np.zeros((dims[0], dims[1], dims[2]))
    ey = np.zeros((dims[0], dims[1], dims[2]))
    ez = np.zeros((dims[0], dims[1], dims[2]))
    dx = np.zeros((dims[0], dims[1], dims[2]))
    dy = np.zeros((dims[0], dims[1], dims[2]))
    dz = np.zeros((dims[0], dims[1], dims[2]))
    hx = np.zeros((dims[0], dims[1], dims[2]))
    hy = np.zeros((dims[0], dims[1], dims[2]))
    hz = np.zeros((dims[0], dims[1], dims[2]))
    gax = np.ones((dims[0], dims[1], dims[2]))
    gay = np.ones((dims[0], dims[1], dims[2]))
    gaz = np.ones((dims[0], dims[1], dims[2]))

    fields = {
        'ex': ex, 'ey': ey, 'ez': ez,
        'dx': dx, 'dy': dy, 'dz': dz,
        'hx': hx, 'hy': hy, 'hz': hz,
        'gax': gax, 'gay': gay, 'gaz': gaz
    }

    return fields


def initResutsToSave(num_steps: list[int], animationData: bool) -> list[dict]:
    plotting_points = []
    if animationData:
        labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        
        for label, num_step in zip(labels[:len(num_steps)], num_steps):
            plotting_points.append(
                {'label': label, 'num_steps': num_step, 
                'data_to_plot': None, 'data_to_save': None,
                }
            )
        z_scales = [0.20, 0.05, 0.05, 0.05]
        for plotting_point, z_scale in zip(plotting_points, z_scales):
            plotting_point['z_scale'] = z_scale
    
    else:
        for num_step in num_steps:
            plotting_points.append(
                {
                    'num_steps': num_step,
                    'data_to_save': None

                }
            )
    return plotting_points


def setGaussianSource(t0: int, time_step: int, spread: float) -> float:
    return exp(-0.5 * ((t0 - time_step) / spread) ** 2)


def setSineSource(freqency: float, timeCellSize: float, time_step: int) -> float:
    return sin(2 * pi * freqency * timeCellSize * time_step)


def specifyDipole(gaz: np.ndarray, gridCenter: list[int]) -> None:
    ic, jc, kc = gridCenter[0], gridCenter[1], gridCenter[2]
    gaz[ic, jc, kc - 10: kc + 10] = 0
    gaz[ic, jc, kc] = 1


def plotSavedResults(dims: list[int], plotting_points: list[dict], nrow: int, ncol: int, zAxisCenter: int) -> None:
    plt.style.use(plt.style.available[6])
    plt.rcParams['font.size'] = 12
    plt.rcParams['grid.color'] = 'midnightblue'
    plt.rcParams['grid.linestyle'] = 'dotted'
    fig = plt.figure(figsize=(8, 6))

    X, Y = np.meshgrid(range(dims[1]), range(dims[0]))

    for subplot_num, plotting_point in enumerate(plotting_points):
        ax = fig.add_subplot(nrow, ncol, subplot_num + 1, projection='3d')
        plot_e_field(ax,
                    plotting_point['data_to_plot'],
                    X,
                    Y,
                    plotting_point['num_steps'],
                    plotting_point['label'],
                    dims,
                    plotting_point['z_scale'],
                    zAxisCenter
                    )
    
    plt.subplots_adjust(bottom=0.05, left=0.10, hspace=0.05)
    plt.show()


def plot_e_field(
        ax, data: np.ndarray, 
        X: np.ndarray, Y: np.ndarray, 
        timestep: int, label: str, 
        dims: list[int], scale: float,
        kc: int
        ):
    """3D plot of E field at a single time step"""
    ax.set_zlim(0, scale)
    ax.view_init(elev=30, azim=-135)
    ax.plot_surface(X, Y, data[:, :, kc], rstride=1, cstride=1, color="white",
                   edgecolor='midnightblue', lw=0.25)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r' $E_Z$', rotation=90, labelpad=10, fontsize=14)
    ax.set_zticks([0, scale / 2, scale])
    ax.set_xlabel('cm')
    ax.set_ylabel('cm')
    ax.set_xticks(np.arange(0, dims[0] + 1, step=20))
    ax.set_yticks(np.arange(0, dims[1] + 1, step=20))
    ax.text2D(0.6, 0.7, "T = {}".format(timestep), transform=ax.transAxes)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    plt.gca().patch.set_facecolor('k')
    ax.text2D(-0.2, 0.8, "({})".format(label), transform=ax.transAxes)
    ax.dist = 11


def getDataToVTK(
        dataNum: int, 
        savedData: list[dict], 
        sourcePos: list[int],
        animationData: bool) -> dict:
    
    if not animationData:
        if dataNum in range(len(savedData)):
            return {
                'fields': [savedData[dataNum]['data_to_save'][0], savedData[dataNum]['data_to_save'][1], savedData[dataNum]['data_to_save'][2]],
                'sourcePos': sourcePos
            }
        else:
            print('Index out of range...')
            return {
                'fields': [savedData[0]['data_to_save'][0], savedData[0]['data_to_save'][1], savedData[0]['data_to_save'][2]],
                'sourcePos': sourcePos
            }
    else:
        data = []
        for d in savedData:
            data.append(
                (d['data_to_save'][0], d['data_to_save'][1], d['data_to_save'][1])
            )
        return {
            'fields': data,
            'sourcePos': sourcePos
        }
    


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
    display = False
    dims = [60, 60, 60]
    animationData = False
    result = fdtd_3D_data(display, dims, animationData)

    if result is not None:
        eField, sourcePos = result[0], result[1]
        ex, ey, ez = eField[0], eField[1], eField[2]
        print(ex)

