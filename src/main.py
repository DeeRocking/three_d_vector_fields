from typing import Tuple

# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkFiltersCore import vtkGlyph3D, vtkContourFilter
from vtkmodules.vtkFiltersSources import vtkArrowSource
from vtkmodules.vtkCommonCore import (
    vtkMath,
    vtkPoints
)
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
)

from commonPipeline import generateRenderingObject
from createDataset import createImageDataSet
from createDataAttributes import createScalarAttrib


def main() -> None:
    colors = vtkNamedColors()
    
    dims = [26, 26, 26]
    origin = [-0.5, -0.5, -0.5]
    sp = 1.0 / 25.0

    vol = createImageDataSet(dims, origin, sp)
    scalars = createScalarAttrib(dims, origin, sp)

    vol.GetPointData().SetScalars(scalars)

    contour = vtkContourFilter()
    contour.SetInputData(vol)
    contour.SetValue(0, 0.0)

    # Visualization
    volMapper = vtkPolyDataMapper()
    volMapper.SetInputConnection(contour.GetOutputPort())
    volMapper.ScalarVisibilityOff()
    
    volActor = vtkActor()
    volActor.SetMapper(volMapper)
    volActor.GetProperty().EdgeVisibilityOn()
    volActor.GetProperty().SetColor(colors.GetColor3d('Salmon'))
    
    renderer, renWin, iren = generateRenderingObject('Vol')
    renderer.AddActor(volActor)
    renderer.SetBackground(colors.GetColor3d('SlateGray'))

    renWin.SetSize(512, 512)

    renWin.Render()

    iren.Start()


if __name__ == "__main__":
    main()
