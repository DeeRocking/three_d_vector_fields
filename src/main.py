from typing import Tuple

# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkFiltersCore import vtkGlyph3D, vtkContourFilter
from vtkmodules.vtkFiltersSources import vtkArrowSource
from vtkmodules.vtkCommonCore import (
    vtkLookupTable
)
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
)

from commonPipeline import generateRenderingObject, generateGlyph3D
from createDataset import createImageDataSet
from createDataAttributes import createVectorAttrib, createVectorAttribFromFDTD


def main() -> None:
    colors = vtkNamedColors()
    windowName: str = '3D vector Field'
    
    N = 14
    dims = [N] * 3
    origin = [-0.5, -0.5, -0.5]
    sp = 15.0 / 25.0

    vectorField = createImageDataSet(dims, origin, sp)
    vectors = createVectorAttribFromFDTD(dims)

    _ = vectorField.GetPointData().SetVectors(vectors)

    glyph3D = generateGlyph3D(vectorField)


    # Visualization
    vectorFieldMapper = vtkPolyDataMapper()
    vectorFieldMapper.SetInputConnection(glyph3D.GetOutputPort())
    
    vectorFieldActor = vtkActor()
    vectorFieldActor.SetMapper(vectorFieldMapper)
    vectorFieldActor.GetProperty().EdgeVisibilityOn()
    vectorFieldActor.GetProperty().SetColor(colors.GetColor3d('Salmon'))
    
    renderer, renWin, iren = generateRenderingObject(windowName)
    renderer.AddActor(vectorFieldActor)
    renderer.SetBackground(colors.GetColor3d('Black'))

    renWin.SetSize(512, 512)

    renWin.Render()

    iren.Start()




if __name__ == "__main__":
    main()
