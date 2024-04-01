import cv2
import os
import sys, getopt
import signal
import time
from edge_impulse_linux.image import ImageImpulseRunner

modelfile = '/home/djolodjolo/Downloads/Downloads/fomo-can-detection-linux-x86_64-v6.eim'

camera_port = None

runner = None

show_camera = True

def now():
    return round(time.time() * 1000)

def now():
    return round(time.time() * 1000)

def get_webcams():
    port_ids = []
    for port in range(5):
        print("Looking for a camera in port %s:" %port)
        camera = cv2.VideoCapture(port)
        if camera.isOpened():
            ret = camera.read()[0]
            if ret:
                backendName =camera.getBackendName()
                w = camera.get(3)
                h = camera.get(4)
                print("Camera %s (%s x %s) found in port %s " %(backendName,h,w, port))
                port_ids.append(port)
            camera.release()
    return port_ids

def sigint_handler(sig, frame):
    print('Interrupted')
    if (runner):
        runner.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, sigint_handler)


print('MODEL: ' + modelfile)



with ImageImpulseRunner(modelfile) as runner:
    try:
        model_info = runner.init()
        print('Loaded runner for "' + model_info['project']['owner'] + ' / ' + model_info['project']['name'] + '"')
        labels = model_info['model_parameters']['labels']
        if camera_port:
            videoCaptureDeviceId = int(args[1])
        else:
            port_ids = get_webcams()
            if len(port_ids) == 0:
                raise Exception('Cannot find any webcams')
            if len(port_ids)> 1:
                raise Exception("Multiple cameras found. Add the camera port ID as a second argument to use to this script")
            videoCaptureDeviceId = int(port_ids[0])

        camera = cv2.VideoCapture(videoCaptureDeviceId)
        ret = camera.read()[0]
        if ret:
            backendName = camera.getBackendName()
            w = camera.get(3)
            h = camera.get(4)
            print("Camera %s (%s x %s) in port %s selected." %(backendName,h,w, videoCaptureDeviceId))
            camera.release()
        else:
            raise Exception("Couldn't initialize selected camera.")

        next_frame = 0 # limit to ~10 fps here
        
        # Define the top of the image and the number of columns
        TOP_Y = 100
        NUM_COLS = 5
        COL_WIDTH = int(w / NUM_COLS)
        # Define the factor of the width/height which determines the threshold
        # for detection of the object's movement between frames:
        DETECT_FACTOR = 1.5

        # Initialize variables
        count = [0] * NUM_COLS
        countsum = 0
        previous_blobs = [[] for _ in range(NUM_COLS)]

        

        for res, img in runner.classifier(videoCaptureDeviceId):
            # Initialize list of current blobs
            current_blobs = [[] for _ in range(NUM_COLS)]
            
            if (next_frame > now()):
                time.sleep((next_frame - now()) / 1000)

            if "bounding_boxes" in res["result"].keys():
                print('Found %d bounding boxes (%d ms.)' % (len(res["result"]["bounding_boxes"]), res['timing']['dsp'] + res['timing']['classification']))
                for bb in res["result"]["bounding_boxes"]:
                    print('\t%s (%.2f): x=%d y=%d w=%d h=%d' % (bb['label'], bb['value'], bb['x'], bb['y'], bb['width'], bb['height']))
                    img = cv2.rectangle(img, (bb['x'], bb['y']), (bb['x'] + bb['width'], bb['y'] + bb['height']), (255, 0, 0), 1)

                        # Check which column the blob is in
                    col = int(bb['x'] / COL_WIDTH)
                    # Check if blob is within DETECT_FACTOR*h of a blob detected in the previous frame and treat as the same object
                    for blob in previous_blobs[col]:
                        print(abs(bb['x'] - blob[0]) < DETECT_FACTOR * (bb['width'] + blob[2]))
                        print(abs(bb['y'] - blob[1]) < DETECT_FACTOR * (bb['height'] + blob[3]))
                        if abs(bb['x'] - blob[0]) < DETECT_FACTOR * (bb['width'] + blob[2]) and abs(bb['y'] - blob[1]) < DETECT_FACTOR * (bb['height'] + blob[3]):
                        # Check this blob has "moved" across the Y threshold
                            if blob[1] >= TOP_Y and bb['y'] < TOP_Y:
                                # Increment count for this column if blob has left the top of the image
                                count[col] += 1
                                countsum += 1
                    # Add current blob to list
                    current_blobs[col].append((bb['x'], bb['y'], bb['width'], bb['height']))
                
            previous_blobs = current_blobs

            if (show_camera):
                im2 = cv2.resize(img, dsize=(800,800))
                cv2.putText(im2, f'{countsum} items passed', (15,750), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                cv2.imshow('edgeimpulse', cv2.cvtColor(im2, cv2.COLOR_RGB2BGR))
                print('Found %d bounding boxes (%d ms.)' % (len(res["result"]["bounding_boxes"]), res['timing']['dsp'] + res['timing']['classification']))

                if cv2.waitKey(1) == ord('q'):
                    break

            next_frame = now() + 100
    finally:
        if (runner):
            runner.stop()
