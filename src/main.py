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
from createDataset import createImageDataSet, createInputData
from createDataAttributes import createVectorAttrib


def main() -> None:
    colors = vtkNamedColors()
    windowName: str = '3D vector Field'
    
    dims = [13, 13, 13]
    origin = [-0.5, -0.5, -0.5]
    sp = 15.0 / 25.0

    vectorField = createImageDataSet(dims, origin, sp)
    vectors = createVectorAttrib(dims, origin, sp)

    _ = vectorField.GetPointData().SetVectors(vectors)

    glyph3D = generateGlyph3D(vectorField)

    # Set a Lookup table to map colors to vector magnitude
    lut = vtkLookupTable()
    lut.SetHueRange(.667, 0.0)
    lut.Build()

    # Visualization
    vectorFieldMapper = vtkPolyDataMapper()
    vectorFieldMapper.SetInputConnection(glyph3D.GetOutputPort())
    vectorFieldMapper.SetScalarRange(0.0, 15.0)
    vectorFieldMapper.SetLookupTable(lut)
    
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
