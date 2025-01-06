from typing import Tuple

from vtkmodules.vtkRenderingCore import (
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)
from vtkmodules.vtkFiltersCore import vtkGlyph3D
from vtkmodules.vtkFiltersSources import vtkArrowSource
from vtkmodules.vtkCommonDataModel import vtkStructuredPoints



def generateRenderingObject(windowName: str) -> Tuple[vtkRenderer, vtkRenderWindow, vtkRenderWindowInteractor]:
    renderer = vtkRenderer()

    renWin = vtkRenderWindow()
    renWin.AddRenderer(renderer)
    renWin.SetWindowName(windowName)

    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    return renderer, renWin, iren


def generateGlyph3D(input_data: vtkStructuredPoints) -> vtkGlyph3D:
    # We use arrow as glyphs
    arrowSource = vtkArrowSource()

    glyph3D = vtkGlyph3D()
    glyph3D.SetSourceConnection(arrowSource.GetOutputPort())
    glyph3D.SetVectorModeToUseVector()  # Use vector for vector field
    glyph3D.SetInputData(input_data)
    glyph3D.SetScaleFactor(1.1)
    glyph3D.Update()

    return glyph3D