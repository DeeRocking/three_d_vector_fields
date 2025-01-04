from typing import Tuple

from vtkmodules.vtkRenderingCore import (
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)



def generateRenderingObject(windowName: str) -> Tuple[vtkRenderer, vtkRenderWindow, vtkRenderWindowInteractor]:
    renderer = vtkRenderer()

    renWin = vtkRenderWindow()
    renWin.AddRenderer(renderer)
    renWin.SetWindowName(windowName)

    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    return renderer, renWin, iren

