from typing import Tuple

from vtkmodules.vtkRenderingCore import (
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)
from vtkmodules.vtkFiltersCore import vtkGlyph3D
from vtkmodules.vtkFiltersSources import vtkArrowSource
from vtkmodules.vtkCommonCore import (
    vtkIdList,
    vtkPoints
)
from vtkmodules.vtkCommonDataModel import (
    VTK_POLYHEDRON,
    vtkUnstructuredGrid,
    vtkStructuredPoints
)
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper
from vtkmodules.vtkFiltersModeling import vtkOutlineFilter
from vtkmodules.vtkCommonColor import vtkNamedColors




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


def generatePlateSource(sourcePos: list[int], dimensions: list[float]) -> vtkUnstructuredGrid:
    points = vtkPoints()

    x, y, z = sourcePos[0], sourcePos[1], sourcePos[2]
    length, width, height = dimensions[0], dimensions[1], dimensions[2]

    points.InsertNextPoint(x - length / 2, y - width / 2, z - height / 2)
    points.InsertNextPoint(x + length / 2, y - width / 2, z - height / 2)
    points.InsertNextPoint(x + length / 2, y + width / 2, z - height / 2)
    points.InsertNextPoint(x - length / 2, y + width / 2, z - height / 2)
    points.InsertNextPoint(x - length / 2, y - width / 2, z + height / 2)
    points.InsertNextPoint(x + length / 2, y - width / 2, z + height / 2)
    points.InsertNextPoint(x + length / 2, y + width / 2, z + height / 2)
    points.InsertNextPoint(x - length / 2, y + width / 2, z + height / 2)

    # These are the point ids corresponding to each face.
    faces = [[0, 3, 2, 1], [0, 4, 7, 3], [4, 5, 6, 7], [5, 1, 2, 6], [0, 1, 5, 4], [2, 3, 7, 6]]
    faceId = vtkIdList()
    faceId.InsertNextId(6)  # Six faces make up the cell.
    for face in faces:
        faceId.InsertNextId(len(face))  # The number of points in the face.
        [faceId.InsertNextId(i) for i in face]

    ugrid = vtkUnstructuredGrid()
    ugrid.SetPoints(points)
    ugrid.InsertNextCell(VTK_POLYHEDRON, faceId)

    return ugrid


def problemSpaceOutline(source: vtkGlyph3D) -> vtkActor:
    outline = vtkOutlineFilter()
    outline.SetInputConnection(source.GetOutputPort())
    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(outline.GetOutputPort())

    actor = vtkActor()
    actor.SetMapper(mapper)
    
    colors = vtkNamedColors()

    actor.GetProperty().SetColor(colors.GetColor3d('White'))

    return actor