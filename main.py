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

def put_boxes(box, frame):

    # bounding box
    confidence = math.ceil(box.conf[0]*100)/100
    if confidence >=0.75:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
        #put box in cam
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        #put text in cam
        org = [x1, y1-5]
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (0, 255, 0)
        thickness = 1
        cv2.putText(frame, ("%.2f") % confidence, org, font, fontScale, color, thickness)
    else:
        pass

def get_video(source):
    try:
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    except:
        pass
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    return cap
def main():
    
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
        if success == False:
            cap.release()
            time.sleep(1)
            cap = get_video(0)
        else:
            pass
                        
        if plc.get_cpu_state() == 'S7CpuStatusUnknown':
            plc_connected = False
            try:
                print("Trying to connect")
                plc.disconnect()
                time.sleep(1)
                plc.connect(ip,0,1)
            except:
                print("Cant connect")
        
        if plc.get_cpu_state() == 'S7CpuStatusRun':
            plc_connected = True
            test_start = PLC_Comm.read_bool(plc,20,2,0)
            

        if success and plc_connected:
            print("CONNECTED")
            results = model(frame, stream=True, iou = 0.1, imgsz=640)
            #results = openvino_model(frame, imgsz = 640, iou = 0.1)
            for r in results:
                boxes = r.boxes
                handle_count = 0
                cover_count = 0
                
                for box in boxes:
                    label = r.names[box.cls[0].item()] 
                    put_boxes(box, frame)
                
                    if label == "handle":
                        handle_count +=1    
                    elif label == "cover":
                        cover_count +=1      

                if test_start:
                    recete_no = PLC_Comm.read_int(plc,20,0)

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
        cv2.imshow("handle_cover_detection", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()