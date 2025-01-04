"""
Simulation in free space
"""
from math import exp
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


SPATIAL_DIMENSION: int = 200

# Pulse parameters
domainCenter = SPATIAL_DIMENSION // 2
t0 = 40
spread = 12

TEMPORAL_DIMENSION: int = 100



def main() -> None:
    ex, hy = initializeFields(SPATIAL_DIMENSION)

    GaussianSourceProfile = {
        't0': t0,
        'spread': spread,
        'position': domainCenter
    }
    
    mainFDTDLoop(ex, hy, SPATIAL_DIMENSION, TEMPORAL_DIMENSION, GaussianSourceProfile)
    
    time_steps = np.arange(TEMPORAL_DIMENSION)
    plotResults(ex, hy, SPATIAL_DIMENSION, time_steps[99])



def initializeFields(dims: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize the Electric and magnetic fields for a field propagating
    in the z direction. The electric field is directing along the x direction
    and the magnetic field is pointing in the y direction.
    """
    ex = np.zeros(shape=(dims,))
    hy = np.zeros(shape=(dims,))
    return ex, hy


def mainFDTDLoop(eField: np.ndarray, hField: np.ndarray, spatialDim: int, temporalDim: int, sourceProfile: dict) -> None:
    for time_step in range(1, temporalDim + 1):

        # Calculate the Ex field
        for k in range(1, spatialDim):
            eField[k] = eField[k] + 0.5 * (hField[k-1] - hField[k])

        # Put a Gaussian pulse in the middle of the problem domain
        pulse = exp(-0.5 * ((sourceProfile['t0'] - time_step) / sourceProfile['spread']) ** 2)
        eField[sourceProfile['position']] = pulse

        # Calculate the Hy field
        for k in range(spatialDim -1):
            hField[k] = hField[k] + 0.5 * (eField[k] - eField[k+1])


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


if __name__ == "__main__":
    main()