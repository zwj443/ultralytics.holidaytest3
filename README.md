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
想要获得三维空间中方向，当前信息不够。yolo的物体检测是，2D的，用边界框圈出，不知道物体的物体的实际物理尺寸，距离摄像头的真实距离在三维空间中的朝向（偏航、俯仰、横滚）和三维位置坐标

