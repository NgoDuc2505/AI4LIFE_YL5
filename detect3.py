import torch

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")
model.classes = [0, 2]  # person and car

# Image


# Inference
results = model(
    "D:/PersonProject/Project_with_Py/Model4days/yolov5/yolov5/testCase.MP4"
)
results.print()  # or .show(), .save()
