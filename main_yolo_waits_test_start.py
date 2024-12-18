import cv2, math
from ultralytics import YOLO
import snap7
import os, sys
import multiprocessing
import time
import keyboard
import PLC_Comm

def resource_path(relative):

    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative)

def put_boxes(box, frame, label):

    # bounding box
    confidence = math.ceil(box.conf[0]*100)/100

    x1, y1, x2, y2 = box.xyxy[0]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

    #If detected object isn't in recipe make the boxes and texts red
    if label in valid_classes:
        color = (0,255,0)
    else:
        color = (0,0,255)
    
    #put box in cam
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

    #put text in cam
    org = [x1, y1-5]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    thickness = 1
    cv2.putText(frame, ("%.2f") % confidence, org, font, fontScale, color, thickness)

def put_info(frame):
    if plc_connected:
        text = "PLC Baglanti: OK"
    else:
        text = "PLC Baglanti: NOK"

    x,y,w,h = 0,0,340,75

    # Draw black background rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,0), -1)
    # Add text
    cv2.putText(frame, text, (x + int(w/10), y + int(h/2)+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

def get_video(source):
    try:
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except:
        pass
    return cap

def main():
    global plc_connected, valid_classes
    ip = "10.20.17.231"
    plc_connected = False
    test_start = False
    plc = snap7.client.Client()
    cap = get_video(0)
    pt = resource_path('handle_cover_nano_400_v3_133.pt')
    model = YOLO(pt)
    
    # pt2 =resource_path("handle_cover_small_v2_openvino_model/")
    # openvino_model = YOLO(pt2)

    while True:

        success, frame = cap.read()
        if success:
            frame = cv2.resize(frame, (1728,972))
            print(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
             
        else:
            cap.release()
            time.sleep(1)
            cap = get_video(0)
                        
        if plc.get_cpu_state() == 'S7CpuStatusUnknown':
            plc_connected = False
            try:
                plc.disconnect()
                time.sleep(1)
                plc.connect(ip,0,1)
            except:
                pass
        
        if plc.get_cpu_state() == 'S7CpuStatusRun':
            plc_connected = True
            test_start = PLC_Comm.read_bool(plc,20,2,0)
            

        if success == True:

            if plc_connected and test_start:
                results = model(frame, stream=True, iou = 0.1, imgsz=640, conf = 0.75)
                recete_no = PLC_Comm.read_int(plc,20,0)
                #results = openvino_model(frame, imgsz = 640, iou = 0.1)
                valid_classes = ['handle', 'cover'] if (recete_no == 2 or recete_no == 8) else ['cover']
                
                for r in results:
                    boxes = r.boxes
                    handle_count = 0
                    cover_count = 0
                    
                    for box in boxes:
                        label = r.names[box.cls[0].item()] 
                        put_boxes(box, frame, label)
                    
                        if label == "handle":
                            handle_count +=1    
                        elif label == "cover":
                            cover_count +=1    
                        
                if recete_no == 2 or recete_no == 8:       
                    if handle_count == 2 and cover_count == 2:                   
                        PLC_Comm.write_bool(plc,20,10,0,True)
                    else:
                        PLC_Comm.write_bool(plc, 20,10,0,False)
                        
                elif recete_no == 6:                        
                    if cover_count == 2 and handle_count == 0:                    
                        PLC_Comm.write_bool(plc,20,10,0,True)
                    else:                    
                        PLC_Comm.write_bool(plc,20,10,0,False)
            else:
                pass

        
        frame = cv2.resize(frame, (1728, 972))
        put_info(frame)
        cv2.imshow("handle_cover_detection", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()