from ultralytics import YOLO
from swanlab.integration.ultralytics import add_swanlab_callback
import swanlab

def main():
    swanlab.init(project="python_crop", experiment_name="YOLOv11n",)
    model = YOLO("yolo11n.pt")
    add_swanlab_callback(model)
    # 将下面的路径替换成你的绝对路径
    model.train(data="C:\\Users\\XQINM\\Desktop\\soilKnow\\python_crop\\datasets\\data.yaml", epochs=5, imgsz=512, batch=8)

if __name__ == "__main__":
    main()