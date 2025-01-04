from vtkmodules.vtkCommonDataModel import vtkStructuredPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkCommonCore import vtkDoubleArray

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
