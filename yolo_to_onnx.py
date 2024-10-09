from ultralytics import YOLO

model = YOLO('../model/plate1.pt')  

model.export(format='onnx')
