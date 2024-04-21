import cv2
import sys
from edge_impulse_linux.image import ImageImpulseRunner

modelfile = '/home/djolodjolo/FOMO-cans-detection-linux/fomo-can-detection-linux-aarch64-v12.eim'

def get_first_available_camera():
    for port in range(5):
        camera = cv2.VideoCapture(port)
        if camera.isOpened():
            camera.release()
            return port
    return None

camera_port = get_first_available_camera()
if camera_port is None:
    sys.exit("No available camera found.")

with ImageImpulseRunner(modelfile) as runner:
    model_info = runner.init()
    print('Model loaded from project:', model_info['project']['owner'], '/', model_info['project']['name'])

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 30.0, (640, 640))

    for res, img in runner.classifier(camera_port):
        if "bounding_boxes" in res["result"]:
            for bb in res["result"]["bounding_boxes"]:
                
                #centroids
                x = bb['x'] + bb['width'] / 2
                y = bb['y'] + bb['height'] / 2

                cv2.circle(img, (int(x), int(y)), 5, (255, 0, 0), -1) 
                cv2.putText(img, bb['label'], (bb['x'], bb['y'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  

        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (640, 640))

        out.write(img_resized)

        cv2.imshow('Frame', img_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    runner.stop()
    out.release()
