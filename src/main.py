from typing import Tuple

# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkDataSetMapper
)

from commonPipeline import (
    generateRenderingObject,
    generateGlyph3D, 
    generatePlateSource,
    problemSpaceOutline,
    generateAxesActor
)
from createDataset import createImageDataSet, getDataFromFDTD
from createDataAttributes import createVectorAttribFromFDTD
from animation import vtkTimerCallback


def main() -> None:
    colors = vtkNamedColors()
    windowName: str = '3D vector Field'
    
    N = 14
    dims = [N] * 3
    origin = [-0.5, -0.5, -0.5]
    sp = 25.0 / 25.0

    vectorField = createImageDataSet(dims, origin, sp)
    animationData = False

    data = getDataFromFDTD(dims, 3, animationData)
    if data is not None:
        if not animationData:
            fields = data['fields']
        else:
            fieldsList = data['fields']
            fields = fieldsList[0]
            
        sourcePos = data['sourcePos']
        vectors = createVectorAttribFromFDTD(dims, (fields[0], fields[1], fields[2]))


        _ = vectorField.GetPointData().SetVectors(vectors)

        glyph3D = generateGlyph3D(vectorField)


        # Visualization
        vectorFieldMapper = vtkPolyDataMapper()
        vectorFieldMapper.SetInputConnection(glyph3D.GetOutputPort())
        
        vectorFieldActor = vtkActor()
        vectorFieldActor.SetMapper(vectorFieldMapper)
        vectorFieldActor.GetProperty().EdgeVisibilityOn()
        vectorFieldActor.GetProperty().SetColor(colors.GetColor3d('Salmon'))

        length, width, heigth = 0.5, 0.5, 3.0
        plate = generatePlateSource(sourcePos, [length, width, heigth])

        plateMapper = vtkDataSetMapper()
        plateMapper.SetInputData(plate)

        plateActor = vtkActor()
        plateActor.SetMapper(plateMapper)
        plateActor.GetProperty().SetColor(colors.GetColor3d('Silver'))

        outlineActor = problemSpaceOutline(glyph3D)

        axisActor = generateAxesActor()
        
        renderer, renWin, iren = generateRenderingObject(windowName)
        renderer.AddActor(vectorFieldActor)
        renderer.AddActor(plateActor)
        renderer.AddActor(outlineActor)
        renderer.AddActor(axisActor)
        renderer.SetBackground(colors.GetColor3d('Black'))

        renWin.SetSize(1200, 800)

        if animationData:
            iren.Initialize()

            cb = vtkTimerCallback(
                len(fieldsList),
                dims,
                vectorFieldActor,
                vectorField,
                fieldsList,
                iren
            )
            
            iren.AddObserver('TimerEvent', cb.execute)
            cb.timerId = iren.CreateRepeatingTimer(300)

        iren.Initialize()
        renWin.Render()

        iren.Start()



if __name__ == "__main__":
    main()
