import torch 
import os
import cv2 as cv
import numpy as np

cap= cv.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
Labels = { 0 : 'Headlights',}
def score_frame(frame, model):
    frame = [torch.tensor(frame)]
    results = model(frame)
    labels = results.xyxyn[0][:, -1].numpy()
    cord = results.xyxyn[0][:, :-1].numpy()
    return labels, cord

source_path = "D:\PersonProject\Project_with_Py\Model4days\yolov5\yolov5/runs/exp/weights/best.pt"
source_path2 = os.path.join(os.getcwd(), "runs/exp/weights/best.pt")

model = torch.hub.load(".", 'custom', path=source_path2, source="local") 

def argmax(prediction):
    prediction = prediction.cpu()
    prediction = prediction.detach().numpy()
    top_1 = np.argmax(prediction, axis=1)
    score = np.amax(prediction)
    score = '{:6f}'.format(score)
    prediction = top_1[0]
    result = Labels[prediction]
    return result,score

# Images
imgs = "008605.png"  
# Inference
results = model(imgs)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
while True:
    ret, frame = cap.read()
    pred = model(frame)
    pred.print()
    cv.imshow("ASL SIGN DETECTER", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyWindow("ASL SIGN DETECTER")
# Results
results.print()
results.save()


