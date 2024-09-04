import numpy as np
import cv2
import mss
import os

def Initialize():
    global sct
    global display
    global monitor
    global monitor_x1
    global monitor_y1
    global monitor_x2
    global monitor_y2
    global cam
    global cam_library

    sct = mss.mss()
    display = 0
    monitor = sct.monitors[(display + 1)]
    monitor_x1 = monitor["left"]
    monitor_y1 = monitor["top"]
    monitor_x2 = monitor["width"]
    monitor_y2 = monitor["height"]
    cam = None
    cam_library = None

    try:

        if os.name == "nt":

            try:

                from windows_capture import WindowsCapture, Frame, InternalCaptureControl
                capture = WindowsCapture(
                    cursor_capture=False,
                    draw_border=False,
                    monitor_index=display + 1,
                    window_name=None,
                )
                global WindowsCaptureFrame
                global StopWindowsCapture
                StopWindowsCapture = False
                @capture.event
                def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):
                    global WindowsCaptureFrame
                    global StopWindowsCapture
                    WindowsCaptureFrame = frame.convert_to_bgr().frame_buffer.copy()
                    if StopWindowsCapture:
                        StopWindowsCapture = False
                        capture_control.stop()
                @capture.event
                def on_closed():
                    print("Capture Session Closed")
                try:
                    control.stop()
                except:
                    pass
                control = capture.start_free_threaded()

                cam_library = "WindowsCapture"

            except:

                import bettercam
                try:
                    cam.stop()
                except:
                    pass
                try:
                    cam.close()
                except:
                    pass
                try:
                    cam.release()
                except:
                    pass
                try:
                    del cam
                except:
                    pass
                cam = bettercam.create(output_idx=display, output_color="BGR")
                cam.start()
                cam.get_latest_frame()
                cam_library = "BetterCam"

        else:

            cam_library = "MSS"

    except:

        cam_library = "MSS"


def plugin(ImageType:str = "both"):
    """ImageType: "both", "cropped", "full" """

    if cam_library == "WindowsCapture":

        try:

            img = cv2.cvtColor(np.array(WindowsCaptureFrame), cv2.COLOR_BGRA2BGR)
            if ImageType == "both":
                croppedImg = img[monitor_y1:monitor_y2, monitor_x1:monitor_x2]
                return croppedImg, img
            elif ImageType == "cropped":
                croppedImg = img[monitor_y1:monitor_y2, monitor_x1:monitor_x2]
                return croppedImg
            elif ImageType == "full":
                return img
            else:
                croppedImg = img[monitor_y1:monitor_y2, monitor_x1:monitor_x2]
                return croppedImg, img

        except:

            return None if ImageType == "cropped" or ImageType == "full" else (None, None)

    elif cam_library == "BetterCam":

        try:

            if cam == None:
                Initialize()
            img = np.array(cam.get_latest_frame())
            if ImageType == "both":
                croppedImg = img[monitor_y1:monitor_y2, monitor_x1:monitor_x2]
                return croppedImg, img
            elif ImageType == "cropped":
                croppedImg = img[monitor_y1:monitor_y2, monitor_x1:monitor_x2]
                return croppedImg
            elif ImageType == "full":
                return img
            else:
                croppedImg = img[monitor_y1:monitor_y2, monitor_x1:monitor_x2]
                return croppedImg, img

        except:

            return None if ImageType == "cropped" or ImageType == "full" else (None, None)

    elif cam_library == "MSS":

        try:

            fullMonitor = sct.monitors[(display + 1)]
            img = np.array(sct.grab(fullMonitor))
            if ImageType == "both":
                croppedImg = img[monitor_y1:monitor_y2, monitor_x1:monitor_x2]
                return croppedImg, img
            elif ImageType == "cropped":
                croppedImg = img[monitor_y1:monitor_y2, monitor_x1:monitor_x2]
                return croppedImg
            elif ImageType == "full":
                return img
            else:
                croppedImg = img[monitor_y1:monitor_y2, monitor_x1:monitor_x2]
                return croppedImg, img

        except:

            return None if ImageType == "cropped" or ImageType == "full" else (None, None)