import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

print("摄像头初始化成功")
print("开始实时检测，按 'q' 键退出...")

ret, test_frame = cap.read()
frame_height, frame_width = test_frame.shape[:2] if ret else (480, 640)

TARGET_CLASSES = ['cup', 'bottle', 'laptop', 'cell phone']

# 用于跟踪同类物体的计数器
class_counters = {}

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break

        results = model(frame)
        boxes = results[0].boxes

        # 重置计数器
        class_counters = {}

        if len(boxes) > 0:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                if class_name in TARGET_CLASSES:
                    # 更新同类物体计数器
                    if class_name not in class_counters:
                        class_counters[class_name] = 1
                    else:
                        class_counters[class_name] += 1

                    # 获取当前物体的序号
                    object_number = class_counters[class_name]

                    # 大小分析
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height

                    if area > 30000:
                        size_desc = "超大"
                    elif area > 15000:
                        size_desc = "大型"
                    elif area > 5000:
                        size_desc = "中型"
                    else:
                        size_desc = "小型"

                    # 位置分析
                    if center_x < frame_width // 4:
                        horizontal_pos = "最左侧"
                    elif center_x < frame_width // 2:
                        horizontal_pos = "左侧"
                    elif center_x < 3 * frame_width // 4:
                        horizontal_pos = "右侧"
                    else:
                        horizontal_pos = "最右侧"

                    if center_y < frame_height // 4:
                        vertical_pos = "顶部"
                    elif center_y < frame_height // 2:
                        vertical_pos = "中上"
                    elif center_y < 3 * frame_height // 4:
                        vertical_pos = "中下"
                    else:
                        vertical_pos = "底部"

                    position_desc = f"{vertical_pos}{horizontal_pos}"

                    # 颜色分析
                    x1_int, y1_int, x2_int, y2_int = int(x1), int(y1), int(x2), int(y2)
                    color_desc = "颜色未知"
                    if (0 <= x1_int < x2_int <= frame.shape[1] and
                            0 <= y1_int < y2_int <= frame.shape[0]):
                        roi = frame[y1_int:y2_int, x1_int:x2_int]
                        if roi.size > 0:
                            avg_color = np.mean(roi, axis=(0, 1))
                            # 更精细的颜色分类
                            r, g, b = avg_color[2], avg_color[1], avg_color[0]

                            if r > max(g, b) + 30:
                                color_desc = "红色系"
                            elif g > max(r, b) + 30:
                                color_desc = "绿色系"
                            elif b > max(r, g) + 30:
                                color_desc = "蓝色系"
                            elif r > 200 and g > 200 and b > 200:
                                color_desc = "浅色"
                            elif r < 100 and g < 100 and b < 100:
                                color_desc = "深色"
                            else:
                                color_desc = "混合色"

                    # 置信度
                    confidence = float(box.conf[0])

                    # 生成唯一描述符（包含序号）
                    object_description = f"{class_name}{object_number}({position_desc}{size_desc}{color_desc})"

                    print(
                        f"📍 {object_description}: 坐标({center_x},{center_y}) 置信度:{confidence:.2f} 面积:{area:.0f}")

                    # 在画面上标记（显示序号）
                    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                    cv2.putText(frame, f"{class_name}{object_number}",
                                (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, f"({center_x},{center_y})",
                                (center_x + 10, center_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                else:
                    # 对于非关注类别
                    print(f"📌 {class_name}: 中心坐标({center_x},{center_y})")
                    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                    cv2.putText(frame, f"({center_x},{center_y})",
                                (center_x + 10, center_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        annotated_frame = results[0].plot()
        cv2.imshow('YOLOv8实时检测 - 按q退出', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("用户退出")
            break

except KeyboardInterrupt:
    print("程序被中断")
except Exception as e:
    print(f"发生错误: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()














