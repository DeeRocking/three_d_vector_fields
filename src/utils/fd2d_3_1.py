"""2D FDTD, TM program"""

from typing import Tuple
from math import exp

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

    TEMPORAL_DIMENSION = 50


    ez, dz, hx, hy, gaz = initFields([xAxisSize, yAxisSize])
    plotting_points = initResutsToSave([20, 30, 40, 50])

    gaussianSource = {
        'type': 'gaussian',
        't0': t0,
        'spread': spread,
        'position': [xAxisCenter, yAxisCenter]
    }

    mainFDTDLoop(
        [ez, dz, hx, hy, gaz],
        TEMPORAL_DIMENSION,
        [xAxisSize, yAxisSize],
        gaussianSource,
        plotting_points
    )

    if not display:
        return hx, hy
    else:
        plotSavedResults(
            [xAxisSize, yAxisSize],
            plotting_points,
            2,
            2
        )
        return None



def mainFDTDLoop(
        fields: list[np.ndarray], 
        temporalDim: int,
        spatialDims: list[int],
        sourceProfile: dict,
        plotting_points: list[dict]
        ) -> None:
    
    ez, dz, hx, hy, gaz = fields[0], fields[1], fields[2], fields[3], fields[4]

    for time_step in range(1, temporalDim + 1):
        # Calculate Dz
        for j in range(1, spatialDims[1]):
            for i in range(1, spatialDims[0]):
                dz[i, j] = dz[i, j] + 0.5 * (hy[i, j] - hy[i-1, j] - hx[i, j] + hx[i, j-1])
        
        if sourceProfile['type'] == 'gaussian':
            pulse = setGaussianSource(
                sourceProfile['t0'],
                time_step,
                sourceProfile['spread']
            )
        dz[sourceProfile['position'][0], sourceProfile['position'][1]] = pulse

        # Calculate Ez from Dz
        for j in range(1, spatialDims[1]):
            for i in range(1, spatialDims[0]):
                ez[i, j] = gaz[i, j] * dz[i, j]
        
        # Calculate the Hx field
        for j in range(spatialDims[1] - 1):
            for i in range(spatialDims[0] - 1):
                hx[i, j] = hx[i, j] + 0.5 * (ez[i, j] - ez[i, j+1])

        # Calculate the Hx field
        for j in range(spatialDims[1] - 1):
            for i in range(spatialDims[0] - 1):
                hy[i, j] = hy[i, j] + 0.5 * (ez[i + 1, j] - ez[i, j])
        
        # Save the data at certain points for later plotting
        for plotting_point in plotting_points:
            if time_step == plotting_point['num_steps']:
                plotting_point['data_to_plot'] = np.copy(ez)
               



def initFields(dims: list[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ez = np.zeros((dims[0], dims[1]))
    dz = np.zeros((dims[0], dims[1]))
    hx = np.zeros((dims[0], dims[1]))
    hy = np.zeros((dims[0], dims[1]))
    gaz = np.ones((dims[0], dims[1]))

    return ez, dz, hx, hy, gaz


def initResutsToSave(num_steps: list[int]) -> list[dict]:
    labels = ['a', 'b', 'c', 'd']
    plotting_points = []
    for label, num_step in zip(labels, num_steps):
        plotting_points.append(
            {'label': label, 'num_steps': num_step, 'data_to_plot': None}
        )
    return plotting_points


def setGaussianSource(t0: int, time_step: int, spread: float) -> float:
    return exp(-0.5 * ((t0 - time_step) / spread) ** 2)


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



if __name__ == "__main__":
    display = False
    dims = [60, 60]
    result = fdtd_3D_data(display, dims)

    if result is not None:
        hx, hy = result[0], result[1]
        print(hx)

