import json
import cv2
import os

PATH = os.path.dirname(os.path.dirname(__file__)).replace("\\", "/") + "/"
if os.path.exists(f"{PATH}Datasets/Final") == False:
    os.mkdir(f"{PATH}Datasets/Final")

print("Preprocessing data...")

for i, folder in enumerate(os.listdir(f"{PATH}Datasets/Raw")):
    for j, file in enumerate(os.listdir(f"{PATH}Datasets/Raw/{folder}")):
        if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg") and os.path.exists(f"{PATH}Datasets/Preprocessed/{folder}/{file.split('.')[0]}.json"):
            data = json.load(open(f"{PATH}Datasets/Raw/{folder}/{file.split('.')[0]}.json", "r"))
            image = cv2.imread(f"{PATH}Datasets/Raw/{folder}/{file}", cv2.IMREAD_GRAYSCALE)

            x = data["X"]
            y = data["Y"]
            steering = data["Steering"]
            throttle = data["Throttle"]
            brake = data["Brake"]
            speed = data["Speed"]
            rotation = data["Rotation"]

            Name = f"{i}#{j}"
            cv2.imwrite(f"{PATH}Datasets/Final/{Name}#IMAGE.png", image)
            with open(f"{PATH}Datasets/Final/{Name}#INPUT.txt", "w") as f:
                f.write(f"{x}#{y}#{speed}#{rotation}")
            with open(f"{PATH}Datasets/Final/{Name}#OUTPUT.txt", "w") as f:
                f.write(f"{steering}#{throttle}#{brake}")

print("Done!")