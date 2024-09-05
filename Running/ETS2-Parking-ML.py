from TruckSimAPI import scsTelemetry as SCSTelemetry
from SDKController import SCSController
from torchvision import transforms
import ScreenCapture
import numpy as np
import torch
import math
import time
import cv2
import mss
import os

OS = os.name
NAME = "ETS2-Parking-ML"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATH = os.path.dirname(os.path.dirname(__file__)).replace("\\", "/") + "/"

if OS == "nt":
    from ctypes import windll, byref, sizeof, c_int
    import win32gui, win32con

LastScreenCaptureCheck = 0

cv2.namedWindow(NAME, cv2.WINDOW_NORMAL)

_, _, width, height = cv2.getWindowImageRect(NAME)
BACKGROUND = np.zeros((height, width, 3), np.uint8)

cv2.resizeWindow(NAME, width, height)
cv2.imshow(NAME, BACKGROUND)
cv2.waitKey(1)

if OS == "nt":
    HWND = win32gui.FindWindow(None, NAME)
    windll.dwmapi.DwmSetWindowAttribute(HWND, 35, byref(c_int(0x000000)), sizeof(c_int))
    hicon = win32gui.LoadImage(None, f"{PATH}icon.ico", win32con.IMAGE_ICON, 0, 0, win32con.LR_LOADFROMFILE | win32con.LR_DEFAULTSIZE)
    win32gui.SendMessage(HWND, win32con.WM_SETICON, win32con.ICON_SMALL, hicon)
    win32gui.SendMessage(HWND, win32con.WM_SETICON, win32con.ICON_BIG, hicon)

Controller = SCSController()
API = SCSTelemetry()
ScreenCapture.Initialize()


def GetTextSize(text="NONE", text_width=100, max_text_height=100):
    fontscale = 1
    textsize, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontscale, 1)
    width_current_text, height_current_text = textsize
    max_count_current_text = 3
    while width_current_text != text_width or height_current_text > max_text_height:
        fontscale *= min(text_width / textsize[0], max_text_height / textsize[1])
        textsize, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontscale, 1)
        max_count_current_text -= 1
        if max_count_current_text <= 0:
            break
    thickness = round(fontscale * 2)
    if thickness <= 0:
        thickness = 1
    return text, fontscale, thickness, textsize[0], textsize[1]


sct = mss.mss()
def GetScreenDimensions(monitor=1):
    global screen_x, screen_y, screen_width, screen_height
    monitor = sct.monitors[monitor]
    screen_x = monitor["left"]
    screen_y = monitor["top"]
    screen_width = monitor["width"]
    screen_height = monitor["height"]
    return screen_x, screen_y, screen_width, screen_height
GetScreenDimensions()


def GetGamePosition():
    global LastGetGamePosition
    if OS == "nt":
        if LastGetGamePosition[0] + 1 < time.time():
            hwnd = None
            top_windows = []
            window = LastGetGamePosition[1], LastGetGamePosition[2], LastGetGamePosition[3], LastGetGamePosition[4]
            win32gui.EnumWindows(lambda hwnd, top_windows: top_windows.append((hwnd, win32gui.GetWindowText(hwnd))), top_windows)
            for hwnd, window_text in top_windows:
                if "Truck Simulator" in window_text and "Discord" not in window_text:
                    rect = win32gui.GetClientRect(hwnd)
                    tl = win32gui.ClientToScreen(hwnd, (rect[0], rect[1]))
                    br = win32gui.ClientToScreen(hwnd, (rect[2], rect[3]))
                    window = (tl[0], tl[1], br[0] - tl[0], br[1] - tl[1])
                    break
            LastGetGamePosition = time.time(), window[0], window[1], window[0] + window[2], window[1] + window[3]
            return window[0], window[1], window[0] + window[2], window[1] + window[3]
        else:
            return LastGetGamePosition[1], LastGetGamePosition[2], LastGetGamePosition[3], LastGetGamePosition[4]
    else:
        return screen_x, screen_y, screen_x + screen_width, screen_y + screen_height
LastGetGamePosition = 0, screen_x, screen_y, screen_width, screen_height


def GetScreenIndex(x, y):
    with mss.mss() as sct:
        monitors = sct.monitors
    closest_screen_index = None
    closest_distance = float('inf')
    for i, monitor in enumerate(monitors[1:]):
        center_x = (monitor['left'] + monitor['left'] + monitor['width']) // 2
        center_y = (monitor['top'] + monitor['top'] + monitor['height']) // 2
        distance = ((center_x - x) ** 2 + (center_y - y) ** 2) ** 0.5
        if distance < closest_distance:
            closest_screen_index = i + 1
            closest_distance = distance
    return closest_screen_index


def ValidateCaptureArea(monitor, x1, y1, x2, y2):
    monitor = sct.monitors[monitor]
    width, height = monitor["width"], monitor["height"]
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    if x1 == x2:
        if x1 == 0:
            x2 = width - 1
        else:
            x1 = 0
    if y1 == y2:
        if y1 == 0:
            y2 = height - 1
        else:
            y1 = 0
    return x1, y1, x2, y2


def ConvertToScreenCoordinate(x:float, y:float, z:float):

    head_yaw = head_rotation_degrees_x
    head_pitch = head_rotation_degrees_y
    head_roll = head_rotation_degrees_z

    rel_x = x - head_x
    rel_y = y - head_y
    rel_z = z - head_z

    cos_yaw = math.cos(math.radians(-head_yaw))
    sin_yaw = math.sin(math.radians(-head_yaw))
    new_x = rel_x * cos_yaw + rel_z * sin_yaw
    new_z = rel_z * cos_yaw - rel_x * sin_yaw

    cos_pitch = math.cos(math.radians(-head_pitch))
    sin_pitch = math.sin(math.radians(-head_pitch))
    new_y = rel_y * cos_pitch - new_z * sin_pitch
    final_z = new_z * cos_pitch + rel_y * sin_pitch

    cos_roll = math.cos(math.radians(head_roll))
    sin_roll = math.sin(math.radians(head_roll))
    final_x = new_x * cos_roll - new_y * sin_roll
    final_y = new_y * cos_roll + new_x * sin_roll

    if final_z >= 0:
        return None, None, None

    fov_rad = math.radians(80)
    window_distance = (height * (4 / 3) / 2) / math.tan(fov_rad / 2)

    screen_x = (final_x / final_z) * window_distance + width / 2
    screen_y = (final_y / final_z) * window_distance + height / 2

    screen_x = width - screen_x

    distance = math.sqrt((rel_x ** 2) + (rel_y ** 2) + (rel_z ** 2))

    return screen_x, screen_y, distance


MODEL_PATH = ""
for file in os.listdir(f"{PATH}Models"):
    if file.endswith(".pt"):
        MODEL_PATH = f"{PATH}Models/{file}"
        break
if MODEL_PATH == "":
    print("No model found.")
    exit()

print(f"Model: {MODEL_PATH}")

metadata = {"data": []}
model = torch.jit.load(os.path.join(MODEL_PATH), _extra_files=metadata, map_location=DEVICE)
model.eval()

metadata = eval(metadata["data"])
for var in metadata:
    if "classes" in var:
        CLASSES = int(var.split("#")[1])
    if "image_width" in var:
        IMG_WIDTH = int(var.split("#")[1])
    if "image_height" in var:
        IMG_HEIGHT = int(var.split("#")[1])
    if "image_channels" in var:
        IMG_CHANNELS = str(var.split("#")[1])
    if "training_dataset_accuracy" in var:
        print("Training dataset accuracy: " + str(var.split("#")[1]))
    if "validation_dataset_accuracy" in var:
        print("Validation dataset accuracy: " + str(var.split("#")[1]))
    if "val_transform" in var:
        transform = var.replace("\\n", "\n").replace('\\', '').split("#")[1]
        transform_list = []
        transform_parts = transform.strip().split("\n")
        for part in transform_parts[1:-1]:
            part = part.strip()
            if part:
                try:
                    transform_args = []
                    transform_name = part.split("(")[0]
                    if "(" in part:
                        args = part.split("(")[1][:-1].split(",")
                        for arg in args:
                            try:
                                transform_args.append(int(arg.strip()))
                            except ValueError:
                                try:
                                    transform_args.append(float(arg.strip()))
                                except ValueError:
                                    transform_args.append(arg.strip())
                    if transform_name == "ToTensor":
                        transform_list.append(transforms.ToTensor())
                    else:
                        transform_list.append(getattr(transforms, transform_name)(*transform_args))
                except (AttributeError, IndexError, ValueError):
                    print(f"Skipping or failed to create transform: {part}")
        transform = transforms.Compose(transform_list)


while True:
    CurrentTime = time.time()

    data = {}
    data["api"] = API.update()
    frame = ScreenCapture.plugin(ImageType="cropped")

    try:
        _, _, width, height = cv2.getWindowImageRect(NAME)
        if BACKGROUND.shape[0] != height or BACKGROUND.shape[1] != width:
            BACKGROUND = np.zeros((height, width, 3), np.uint8)
    except:
        Controller.close()
        exit()

    if data["api"]["scsValues"]["telemetryPluginRevision"] < 2:
        time.sleep(0.1)
        cv2.imshow(NAME, BACKGROUND)
        cv2.waitKey(1)
        continue

    if type(frame) == type(None):
        continue

    if LastScreenCaptureCheck + 0.5 < CurrentTime:
        game_x1, game_y1, game_x2, game_y2 = GetGamePosition()
        if ScreenCapture.monitor_x1 != game_x1 or ScreenCapture.monitor_y1 != game_y1 or ScreenCapture.monitor_x2 != game_x2 or ScreenCapture.monitor_y2 != game_y2:
            ScreenIndex = GetScreenIndex((game_x1 + game_x2) / 2, (game_y1 + game_y2) / 2)
            if ScreenCapture.display != ScreenIndex - 1:
                if ScreenCapture.cam_library == "WindowsCapture":
                    ScreenCapture.StopWindowsCapture = True
                    while ScreenCapture.StopWindowsCapture == True:
                        time.sleep(0.01)
                ScreenCapture.Initialize()
            ScreenCapture.monitor_x1, ScreenCapture.monitor_y1, ScreenCapture.monitor_x2, ScreenCapture.monitor_y2 = ValidateCaptureArea(ScreenIndex, game_x1, game_y1, game_x2, game_y2)
        LastScreenCaptureCheck = CurrentTime

    width = frame.shape[1]
    height = frame.shape[0]

    if width <= 0 or height <= 0:
        continue


    truck_x = data["api"]["truckPlacement"]["coordinateX"]
    truck_y = data["api"]["truckPlacement"]["coordinateY"]
    truck_z = data["api"]["truckPlacement"]["coordinateZ"]
    truck_rotation_x = data["api"]["truckPlacement"]["rotationX"]
    truck_rotation_y = data["api"]["truckPlacement"]["rotationY"]
    truck_rotation_z = data["api"]["truckPlacement"]["rotationZ"]

    cabin_offset_x = data["api"]["headPlacement"]["cabinOffsetX"] + data["api"]["configVector"]["cabinPositionX"]
    cabin_offset_y = data["api"]["headPlacement"]["cabinOffsetY"] + data["api"]["configVector"]["cabinPositionY"]
    cabin_offset_z = data["api"]["headPlacement"]["cabinOffsetZ"] + data["api"]["configVector"]["cabinPositionZ"]
    cabin_offset_rotation_x = data["api"]["headPlacement"]["cabinOffsetrotationX"]
    cabin_offset_rotation_y = data["api"]["headPlacement"]["cabinOffsetrotationY"]
    cabin_offset_rotation_z = data["api"]["headPlacement"]["cabinOffsetrotationZ"]

    head_offset_x = data["api"]["headPlacement"]["headOffsetX"] + data["api"]["configVector"]["headPositionX"] + cabin_offset_x
    head_offset_y = data["api"]["headPlacement"]["headOffsetY"] + data["api"]["configVector"]["headPositionY"] + cabin_offset_y
    head_offset_z = data["api"]["headPlacement"]["headOffsetZ"] + data["api"]["configVector"]["headPositionZ"] + cabin_offset_z
    head_offset_rotation_x = data["api"]["headPlacement"]["headOffsetrotationX"]
    head_offset_rotation_y = data["api"]["headPlacement"]["headOffsetrotationY"]
    head_offset_rotation_z = data["api"]["headPlacement"]["headOffsetrotationZ"]
    
    truck_rotation_degrees_x = truck_rotation_x * 360
    truck_rotation_radians_x = -math.radians(truck_rotation_degrees_x)

    head_rotation_degrees_x = (truck_rotation_x + cabin_offset_rotation_x + head_offset_rotation_x) * 360
    while head_rotation_degrees_x > 360:
        head_rotation_degrees_x = head_rotation_degrees_x - 360

    head_rotation_degrees_y = (-truck_rotation_y + cabin_offset_rotation_y + head_offset_rotation_y) * 360

    head_rotation_degrees_z = (truck_rotation_z + cabin_offset_rotation_z + head_offset_rotation_z) * 360

    point_x = head_offset_x
    point_y = head_offset_y
    point_z = head_offset_z
    head_x = point_x * math.cos(truck_rotation_radians_x) - point_z * math.sin(truck_rotation_radians_x) + truck_x
    head_y = point_y * math.cos(math.radians(head_rotation_degrees_y)) - point_z * math.sin(math.radians(head_rotation_degrees_y)) + truck_y
    head_z = point_x * math.sin(truck_rotation_radians_x) + point_z * math.cos(truck_rotation_radians_x) + truck_z


    all_valid = True


    offset_x = 1
    offset_y = 0.1
    offset_z = 0.35

    point_x = head_x + offset_x * math.sin(truck_rotation_radians_x) - offset_z * math.cos(truck_rotation_radians_x)
    point_y = head_y + offset_y
    point_z = head_z - offset_x * math.cos(truck_rotation_radians_x) - offset_z * math.sin(truck_rotation_radians_x)

    x1, y1, d1 = ConvertToScreenCoordinate(point_x, point_y, point_z)
    if x1 == None or y1 == None:
        all_valid = False
    else:
        top_left = x1, y1


    offset_x = 1
    offset_y = 0.1
    offset_z = -0.35

    point_x = head_x + offset_x * math.sin(truck_rotation_radians_x) - offset_z * math.cos(truck_rotation_radians_x)
    point_y = head_y + offset_y
    point_z = head_z - offset_x * math.cos(truck_rotation_radians_x) - offset_z * math.sin(truck_rotation_radians_x)

    x1, y1, d1 = ConvertToScreenCoordinate(point_x, point_y, point_z)
    if x1 == None or y1 == None:
        all_valid = False
    else:
        top_right = x1, y1


    offset_x = 1
    offset_y = -0.3
    offset_z = 0.35

    point_x = head_x + offset_x * math.sin(truck_rotation_radians_x) - offset_z * math.cos(truck_rotation_radians_x)
    point_y = head_y + offset_y
    point_z = head_z - offset_x * math.cos(truck_rotation_radians_x) - offset_z * math.sin(truck_rotation_radians_x)

    x1, y1, d1 = ConvertToScreenCoordinate(point_x, point_y, point_z)
    if x1 == None or y1 == None:
        all_valid = False
    else:
        bottom_left = x1, y1


    offset_x = 1
    offset_y = -0.3
    offset_z = -0.35

    point_x = head_x + offset_x * math.sin(truck_rotation_radians_x) - offset_z * math.cos(truck_rotation_radians_x)
    point_y = head_y + offset_y 
    point_z = head_z - offset_x * math.cos(truck_rotation_radians_x) - offset_z * math.sin(truck_rotation_radians_x)

    x1, y1, d1 = ConvertToScreenCoordinate(point_x, point_y, point_z)
    if x1 == None or y1 == None:
        all_valid = False
    else:
        bottom_right = x1, y1


    if all_valid:
        cropped_width = round(max(top_right[0] - top_left[0], bottom_right[0] - bottom_left[0]))
        cropped_height = round(max(bottom_left[1] - top_left[1], bottom_right[1] - top_right[1]))
        src_pts = np.float32([top_left, top_right, bottom_left, bottom_right])
        dst_pts = np.float32([[0, 0], [cropped_width, 0], [0, cropped_height], [cropped_width, cropped_height]])
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        frame = cv2.warpPerspective(frame, matrix, (cropped_width, cropped_height))


    target_x = 10509.855651855469
    target_z = -9956.846969604492
    target_rotation = 0.7545529007911682

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    frame = np.array(frame, dtype=np.float32) / 255.0
    input = truck_x - target_x, truck_z - target_z, target_rotation - truck_rotation_x

    with torch.no_grad():
        output = model(transform(frame).unsqueeze(0).to(DEVICE), torch.as_tensor(input, dtype=torch.float32).unsqueeze(0).to(DEVICE)).tolist()[0]

    print(output)
    #print(truck_x, truck_z, truck_rotation_x)

    Controller.steering = output[0]

    cv2.imshow(NAME, frame)
    cv2.waitKey(1)