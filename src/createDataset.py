from typing import Tuple

import numpy as np
from vtkmodules.vtkCommonDataModel import vtkStructuredPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkCommonCore import vtkDoubleArray


from utils.fd3d_4_1 import fdtd_3D_data

def createImageDataSet(dims: list[int], origin: list[float], sp: float = 1.0 / 25.0) -> vtkStructuredPoints:
    vol = vtkStructuredPoints()
    vol.SetDimensions(dims[0], dims[1], dims[2])
    vol.SetOrigin(origin[0], origin[1], origin[2])
    vol.SetSpacing(sp, sp, sp)

    return vol


def createInputData(inputArray: vtkDoubleArray) -> vtkPolyData:
    inputData = vtkPolyData()

    inputData.ShallowCopy(inputData)

    return inputData


def getDataFromFDTD(dims: list[int], targetIndex: int, animationData: bool) -> dict | None:
    display = False
    fdtdResults = fdtd_3D_data(display, dims, targetIndex, animationData)
    if fdtdResults is not None:
        return fdtdResults
    else:
        return None