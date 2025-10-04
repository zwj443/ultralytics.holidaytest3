import cv2
from ultralytics import YOLO

print("🚀 启动YOLOv8实时检测...")


model = YOLO('yolov8n.pt')
print("✅ 模型加载完成")


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ 无法打开摄像头")
    exit()

print("✅ 摄像头初始化成功")
print("🎥 开始实时检测，按 'q' 键退出...")

try:
    while True:

        ret, frame = cap.read()
        if not ret:
            print("❌ 无法读取摄像头画面")
            break


        results = model(frame)


        boxes = results[0].boxes

        if len(boxes) > 0:
            for i, box in enumerate(boxes):

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()


                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)


                class_id = int(box.cls[0])
                class_name = model.names[class_id]


                print(f"📍 {class_name}: 中心坐标 ({center_x}, {center_y})")


                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"({center_x},{center_y})",
                            (center_x + 10, center_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


        annotated_frame = results[0].plot()
        cv2.imshow('YOLOv8实时检测 - 按q退出', annotated_frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 用户退出")
            break

except KeyboardInterrupt:
    print("👋 程序被中断")
except Exception as e:
    print(f"❌ 发生错误: {e}")
finally:
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ 资源已释放，程序结束")





