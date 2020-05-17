from facenet_pytorch import MTCNN
from PIL import Image
import torch
from imutils.video import FileVideoStream
import cv2
import time
import glob
from tqdm.notebook import tqdm

import numpy as np
from torchvision import transforms
import torch.backends.cudnn as cudnn
from pfld.models.pfld import PFLDInference
from pfld.utils import calculate_pitch_yaw_roll

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

class FastMTCNN(object):
    """Fast MTCNN implementation."""
    
    def __init__(self, stride, resize=1, *args, **kwargs):
        """Constructor for FastMTCNN class.
        
        Arguments:
            stride (int): The detection stride. Faces will be detected every `stride` frames
                and remembered for `stride-1` frames.
        
        Keyword arguments:
            resize (float): Fractional frame scaling. [default: {1}]
            *args: Arguments to pass to the MTCNN constructor. See help(MTCNN).
            **kwargs: Keyword arguments to pass to the MTCNN constructor. See help(MTCNN).
        """
        self.stride = stride
        self.resize = resize
        self.mtcnn = MTCNN(*args, **kwargs)
        
    def __call__(self, frames):
        """Detect faces in frames using strided MTCNN."""
        if self.resize != 1:
            frames = [
                cv2.resize(f, (int(f.shape[1] * self.resize), int(f.shape[0] * self.resize)))
                    for f in frames
            ]
                      
        boxes, probs = self.mtcnn.detect(frames[::self.stride])

        faces = []
        for i, frame in enumerate(frames):
            box_ind = int(i / self.stride)
            if boxes[box_ind] is None:
                continue
            for box in boxes[box_ind]:
                box = [int(b) for b in box]
                faces.append(frame[box[1]:box[3], box[0]:box[2]])
        
        return boxes, faces

fast_mtcnn = FastMTCNN(
    stride=4,
    resize=1,
    margin=14,
    factor=0.6,
    keep_all=True,
    device=device
)


def find_landmarks(img):
    model_path = "./pfld/checkpoint/snapshot/checkpoint.pth.tar"
    checkpoint = torch.load(model_path, map_location=device)
    transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((112, 112)), transforms.ToTensor()])
    plfd_backbone = PFLDInference().to(device)
    plfd_backbone.load_state_dict(checkpoint['plfd_backbone'])
    plfd_backbone.eval()

    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        _, landmarks = plfd_backbone(img.to(device))
    return landmarks

def find_angles(landmarks):
    TRACKED_POINTS = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
    landmarks_2d = landmarks[:, TRACKED_POINTS, :]
    pyr = calculate_pitch_yaw_roll(landmarks_2d)
    return pyr

out = cv2.VideoWriter('depth.avi',cv2.VideoWriter_fourcc(*'DIVX'), 25, (640, 360))
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (0,0,255)
lineType               = 2


def run_detection(fast_mtcnn, filenames):
    frames = []
    frames_processed = 0
    faces_detected = 0
    batch_size = 1
    start = time.time()
    for filename in tqdm(filenames):

        v_cap = FileVideoStream(filename).start()
        v_len = int(v_cap.stream.get(cv2.CAP_PROP_FRAME_COUNT))

        for j in range(v_len):

            frame = v_cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # frame = cv2.flip(frame, flipCode=-1)
            frames.append(frame)
            size = frame.shape
            if len(frames) >= batch_size or j == v_len - 1:

                boxes, faces = fast_mtcnn(frames)

                try:
                    faces[0]
                    landmarks = find_landmarks(faces[0])
                    landmarks = landmarks.cpu().numpy()
                    landmarks = landmarks.reshape(landmarks.shape[0], -1, 2) # landmark
                    p, y, r = map(int,find_angles(landmarks))
                    pyr = 'PITCH: ' + str(p) + ',' + ' YAW: ' + str(y) + ',' + ' ROLL: ' + str(r)
                    pre_landmark = landmarks[0] * [224, 224]
                except:
                    pre_landmark = np.zeros((98, 2))
                    pyr = 'NO DATA'
 
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if boxes[0] is not None:
                    for box in boxes[0]:
                        box = [int(b) for b in box]
                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                for (x, y) in pre_landmark.astype(np.int32):
                    cv2.circle(frame, (x, y), 3, (255,0,0),-1)

                cv2.putText(frame, pyr,
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)
            
                frame = cv2.resize(frame, (640, 360))
                out.write(frame)
                cv2.imshow('0', frame)
                if cv2.waitKey(10) == 27:
                    out.release()
                    break
                frames_processed += len(frames)
                faces_detected += len(faces)
                frames = []

                print(
                    f'Frames per second: {frames_processed / (time.time() - start):.3f},',
                    f'faces detected: {faces_detected}\r',
                    end=''
                )

        v_cap.stop()

run_detection(fast_mtcnn, ['IMG_6364.MOV'])
out.release()