# ultralytics.holidayTest3
配置环境：配置了ultralytics12.9，miniconda,pytorch，然后下载YOLOv8官方预训练权重并验证是否安装成功。
运行程序：Python依赖包
           ultralytics>=8.0.0
           opencv-python>=4.5.0
           torch>=1.7.0
         创建虚拟环境
         python -m venv .venv
         Win激活.venv\Scripts\activate
         MacOS/Linux激活source .venv/bin/activate
         安装依赖包 pip install ultralytics opencv-python
         终端运行 python camera_detection.py

实现步骤：使用YOLO函数-打开摄像头-计算中心点并打印

思考题 对于同类物体，添加了颜色大小和空间识别

