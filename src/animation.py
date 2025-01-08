from typing import Tuple, Optional

import numpy as np
from vtkmodules.vtkCommonDataModel import vtkStructuredPoints
from vtkmodules.vtkRenderingCore import (
    vtkRenderWindowInteractor,
    vtkActor,
    vtkPolyDataMapper
)

from createDataAttributes import createVectorAttribFromFDTD
from commonPipeline import generateGlyph3D


class vtkTimerCallback():
    def __init__(self, 
                 steps: int, 
                 spatialDims: list[int],
                 vectorFieldActor: vtkActor,
                 vectorField: vtkStructuredPoints,
                 fields: list[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                 iren: vtkRenderWindowInteractor
                 ) -> None:
        
        self.timer_count = 0
        self.steps = steps
        self.vectorFieldActor = vectorFieldActor
        self.vectorField = vectorField
        self.fields = fields
        self.iren = iren
        self.timerId = 0
        self.dims = spatialDims

    def execute(self, obj:vtkRenderWindowInteractor, event) -> None:
        step = 0
        while step < self.steps:
            print(f"frame N°{step}/{self.steps}")
            currentFields = self.fields[step]
            vectors = createVectorAttribFromFDTD(self.dims, (currentFields[0], currentFields[1], currentFields[2]))
            _ = self.vectorField.GetPointData().SetVectors(vectors)

            glyph3D = generateGlyph3D(self.vectorField)

            vectorFieldMapper = vtkPolyDataMapper()
            vectorFieldMapper.SetInputConnection(glyph3D.GetOutputPort())
            self.vectorFieldActor.SetMapper(vectorFieldMapper)

            self.iren = obj
            self.iren.GetRenderWindow().Render()
            self.timer_count += 1
            step += 1
        if self.timerId:
            self.iren.DestroyTimer(self.timerId)