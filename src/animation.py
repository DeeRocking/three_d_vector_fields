from typing import Tuple, Optional

import numpy as np
from vtkmodules.vtkCommonDataModel import vtkStructuredPoints
from vtkmodules.vtkRenderingCore import vtkRenderWindowInteractor

from createDataAttributes import createVectorAttribFromFDTD

class vtkTimerCallback():
    def __init__(self, 
                 steps: int, 
                 vectorField: vtkStructuredPoints,
                 fields: list[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                 iren: vtkRenderWindowInteractor
                 ) -> None:
        
        self.timer_count = 0
        self.steps = steps
        self.vectorField = vectorField
        self.fields = fields
        self.iren = iren
        self.timerId = 0
        self.dims = [steps] * 3

    def execute(self, obj:vtkRenderWindowInteractor, event) -> None:
        step = 0
        while step < self.steps:
            print(self.steps)
            currentFields = self.fields[step]
            vectors = createVectorAttribFromFDTD(self.dims, (currentFields[0], currentFields[1], currentFields[2]))
            self.vectorField.GetPointData().SetVectors(vectors)
            iren = obj
            iren.GetRenderWindow().Render()
            self.timer_count += 1
            step += 1
        if self.timerId:
            iren.DestroyTimer(self.timerId)