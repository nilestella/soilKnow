from io import BytesIO

import requests
import requests.exceptions
from PIL import Image
from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO('./runs/detect/train6/weights/best.pt')
model_dis = YOLO('./runs/detect/train6/weights/best_dis.pt')

class ImageRequest(BaseModel):
    image_url: str


@app.route('/crop/detect', methods=['POST'])
def detect():
    try:
        # 获取并验证请求数据
        data = request.get_json()
        if not data or 'image_url' not in data:
            return jsonify({
                "code": 400,
                "success": False,
                "message": "缺少image_url参数",
                "detections": []
            })

        # 验证请求数据
        try:
            img_request = ImageRequest(**data)
        except ValidationError as e:
            return jsonify({
                "code": 400,
                "success": False,
                "message": f"参数验证失败: {str(e)}",
                "detections": []
            })

        # 下载图片
        response = requests.get(img_request.image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))

        # 进行预测
        results = model.predict(
            source=image,
            conf=0.25,
            iou=0.5,
            imgsz=540
        )
        results_dis = model_dis.predict(
            source=image,
            conf=0.25,
            iou=0.5,
            imgsz=540
        )
        # 处理检测结果
        detections = []
        classNames = {"jute": "黄麻", "maize": "玉米", "rice": "水稻", "sugarcane": "甘蔗", "wheat": "小麦"}
        classNames_dis = {"Early Blight":"早疫病","Healthy":"健康","Late Blight":"晚疫病","Leaf Miner":"叶蛾","Leaf Mold":"叶霉病","Mosaic Virus":"马赛克病毒","Septoria":"镰刀菌病","Spider Mites":"螨虫","Yellow Leaf Curl Virus":"黄叶卷曲病毒"}
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                class_name = result.names[class_id]
                detections.append({
                    "class_name": classNames.get(class_name, class_name),
                })
        for result_dis in results_dis:
            for box in result_dis.boxes:
                class_id = int(box.cls)
                class_name = result_dis.names[class_id]
                detections.append({
                    "disease_name": classNames_dis.get(class_name, class_name),
                })
        if detections:
            return jsonify({
                "code": 200,
                "success": True,
                "message": "检测成功",
                "detections": detections,
            })
        else:
            return jsonify({
                "code": 200,
                "success": False,
                "message": "未检测到目标，请重新上传或自定义选择",
                "detections": []
            })

    except requests.exceptions.RequestException as e:
        return jsonify({
            "code": 400,
            "success": False,
            "message": f"图片下载失败: {str(e)}",
            "detections": []
        })
    except Exception as e:
        return jsonify({
            "code": 500,
            "success": False,
            "message": f"处理图片失败: {str(e)}",
            "detections": []
        })


if __name__ == '__main__':
    app.run(debug=True)