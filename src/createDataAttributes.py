import math

from vtkmodules.vtkCommonCore import vtkDoubleArray


def createScalarAttrib(dims: list[int], origin: list[float], sp: float = 1.0 / 25.0) -> vtkDoubleArray:
    scalars = vtkDoubleArray()
    scalars.SetNumberOfComponents(1)
    scalars.SetNumberOfTuples(dims[0] * dims[1] * dims[2])
    
    for k in range(0, dims[2]):
        z = origin[2] + k * sp
        kOffset = k * dims[1] * dims[0]
        for j in range(0, dims[1]):
            y = origin[1] + j * sp
            jOffset = j * dims[0]
            for i in range(0, dims[0]):
                x = origin[0] + i * sp
                s = x * x + y * y + z * z - (0.4 * 0.4)
                offset = i + jOffset + kOffset
                scalars.InsertTuple1(offset, s)
    
    return scalars


def createVectorAttrib(dims: list[int], origin: list[float], sp: float = 1.0 / 25.0) -> vtkDoubleArray:
    vectors = vtkDoubleArray()
    vectors.SetNumberOfComponents(3)
    vectors.SetNumberOfTuples(dims[0] * dims[1] * dims[2])

    for k in range(0, dims[2]):
        z = origin[2] + k * sp
        kOffset = k * dims[1] * dims[0]
        for j in range(0, dims[1]):
            y = origin[1] + j * sp
            jOffset = j * dims[0]
            for i in range(0, dims[0]):
                x = origin[0] + i * sp
                # spatial dependence function here
                x = x * math.cos(x)
                y = y * math.sin(y)
                offset = i + jOffset + kOffset
                vectors.InsertTuple3(offset, x, y, z)

    return vectors