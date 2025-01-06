"""
Simulation in free space
"""
from math import exp, sin
from typing import Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt


SPATIAL_DIMENSION: int = 200

# Pulse parameters
domainCenter = SPATIAL_DIMENSION // 2
t0 = 40
spread = 12

TEMPORAL_DIMENSION: int = 500

epsilon = 4.0
relativEpsilon = 8.854e-12
sigma = 0.04

ddx = 0.01
dt = ddx / 6e8
freq_in = 700e6




def main() -> None:
    ex, hy = initializeFields(SPATIAL_DIMENSION)
    ca, cb = createDielectricProfile(SPATIAL_DIMENSION, 100, epsilon, relativEpsilon, sigma, dt)

    GaussianSourceProfile = {
        'type': 'gaussian',
        't0': t0,
        'spread': spread,
        'position': domainCenter
    }
    sinusoidalProfile = {
        'type': 'sine',
        'freq_in': freq_in,
        'dt': dt
    }
    boundary = {
        'low': [0.0, 0.0],
        'high': [0.0, 0.0]
    }
    plotting_points = [
        {'num_steps': 500, 'data_to_plot': None, 'label': 'FDTD cells'}
    ]
    
    mainFDTDLoop(
        ex, hy,
        SPATIAL_DIMENSION, TEMPORAL_DIMENSION,
        sinusoidalProfile, boundary, plotting_points,
        (ca, cb))
    
    # time_steps = np.arange(TEMPORAL_DIMENSION)
    # plotResults(ex, hy, SPATIAL_DIMENSION, time_steps[50])
    
    # Plot the E field at each of the time steps
    plotSavedResults(plotting_points, cb, epsilon, sigma, len(plotting_points))



def initializeFields(dims: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize the Electric and magnetic fields for a field propagating
    in the z direction. The electric field is directing along the x direction
    and the magnetic field is pointing in the y direction.
    """
    ex = np.zeros(shape=(dims,))
    hy = np.zeros(shape=(dims,))
    print("eField and hField initialized!")
    return ex, hy


def mainFDTDLoop(eField: np.ndarray, hField: np.ndarray,
    spatialDim: int, temporalDim: int, sourceProfile: dict,
    boundary: Dict[str, list[float]], plotting_points: list[dict],
    dielectricMedium: Tuple[np.ndarray, np.ndarray]
    ) -> None:

    preFactor = 0.5
    ca, cb = dielectricMedium[0], dielectricMedium[0] 
    for time_step in range(1, temporalDim + 1):

        # Calculate the Ex field
        for k in range(1, spatialDim):
            eField[k] = ca[k] * eField[k] + cb[k] * (hField[k-1] - hField[k])

        # Put the source in a soft hard maner
        if sourceProfile['type'] == 'gaussian':
            pulse = exp(-0.5 * ((sourceProfile['t0'] - time_step) / sourceProfile['spread']) ** 2)
        elif sourceProfile['type'] == 'sine':
            pulse = sin(2 * np.pi * sourceProfile['freq_in'] * sourceProfile['dt'] * time_step)
        eField[5] = pulse + eField[5]

        # Absorbing Boundary conditions
        setBoundaryConditions(eField, boundary['low'], boundary['high'])

        # Calculate the Hy field
        for k in range(spatialDim -1):
            hField[k] = hField[k] + preFactor * (eField[k] - eField[k+1])

        saveField(plotting_points, time_step, eField, hField)


def createDielectricProfile(
    spaceDim: int, 
    mediumStart: int, 
    epsilon: float,
    relativPermitivity: float,
    conductivity: float, 
    temporalSizeStep: float) -> Tuple[np.ndarray, np.ndarray]:

    cb = np.ones((spaceDim,), dtype=np.float64)
    ca = np.ones((spaceDim,), dtype=np.float64)
    for i in range(spaceDim):
        cb[i] = 0.5 * cb[i]
    
    eaf = temporalSizeStep * conductivity / (2 * relativPermitivity * epsilon)
    ca[mediumStart:] = (1 - eaf) / (1 + eaf)
    cb[mediumStart:] = 0.5 / (epsilon * (1 + eaf))

    return ca, cb



def saveField(plotting_points: list[dict], time_step: int, eField: np.ndarray, hField: np.ndarray) -> None:
    for plotting_point in plotting_points:
        if time_step == plotting_point['num_steps']:
            plotting_point['data_to_plot'] = (np.copy(eField), np.copy(hField))



def setBoundaryConditions(eField: np.ndarray, boundary_low: list[float], boundary_high: list[float]) -> None:
    eField[0] = boundary_low.pop(0)
    boundary_low.append(eField[1])

    dim = eField.size
    eField[dim - 1] = boundary_high.pop(0)
    boundary_high.append(eField[dim - 2])


def plotResults(eField: np.ndarray, hField: np.ndarray, spatialDim: int, time_step: int) -> None:
    plt.style.use(plt.style.available[6])
    plt.rcParams['font.size'] = 12
    plt.figure(figsize=(8, 3.5))

    plt.subplot(211)
    plt.plot(eField, color='darkgreen', linewidth=1)
    plt.ylabel('E$_x$', fontsize=14)
    plt.xticks(np.arange(0, spatialDim + 1, step=20))
    plt.xlim(0, spatialDim)
    plt.yticks(np.arange(-1, 1.2, step=1))
    plt.text(100, 0.5, 'T = {}'.format(time_step + 1))

    plt.subplot(212)
    plt.plot(hField, color='darkgreen', linewidth=1)
    plt.ylabel('H$_y$', fontsize=14)
    plt.xlabel('FDTD cells')
    plt.xticks(np.arange(0, spatialDim + 1, step=20))
    plt.xlim(0, spatialDim)
    plt.yticks(np.arange(-1, 1.2, step=1))
    plt.ylim(-1.2, 1.2)

    plt.subplots_adjust(bottom=0.2, hspace=0.45)
    plt.show()


def plotSavedResults(plotting_points: list[dict], cb: np.ndarray, epsilon: float, sigma: float,numFig: int) -> None:
    plt.style.use(plt.style.available[6])
    plt.rcParams['font.size'] = 12
    fig = plt.figure(figsize=(8, 5.25))

    for subplot_num, plotting_point in enumerate(plotting_points):
        ax = fig.add_subplot(numFig, 1, subplot_num + 1)
        plotEField(plotting_point['data_to_plot'][0], 
                   plotting_point['num_steps'],
                   plotting_point['label'],
                   cb,
                   epsilon,
                   sigma
        )
    
    plt.tight_layout()
    plt.show()

def plotEField(data:np.ndarray, time_step: int, label: str, cb: np.ndarray, epsilon: float, sigma: float) -> None:
    plt.plot(data, color='firebrick', lw=1)
    plt.xticks(np.arange(0, data.size + 1, step=20))
    plt.xlim(0, data.size)
    plt.yticks(np.arange(-1, 1.2, step=1))
    plt.text(50, 0.5, 'T = {}'.format(time_step))
    plt.plot((0.5 / cb -1) / 3, 'g--', lw=0.75)
    plt.text(170, 0.5, 'Eps = {}'.format(epsilon))
    plt.text(170, -0.5, 'Cond = {}'.format(sigma))
    plt.xlabel('{}'.format(label))




if __name__ == "__main__":
    main()