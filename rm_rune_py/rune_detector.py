import cv2
import numpy as np
import onnxruntime
import time
from collections import deque

# 定义类别名称，根据用户需求，模型有两个类别：0和1
class_names = ['0', '1']

# 为每个类别创建颜色
rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(class_names), 3))

# 非极大值抑制函数
def nms(boxes, scores, iou_threshold):
    # 按分数排序
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # 选择最后一个框
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # 计算选中框与其余框的IoU
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # 移除IoU大于阈值的框
        keep_indices = np.where(ious < iou_threshold)[0]
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes

# 多类别非极大值抑制
def multiclass_nms(boxes, scores, class_ids, iou_threshold):
    unique_class_ids = np.unique(class_ids)

    keep_boxes = []
    for class_id in unique_class_ids:
        class_indices = np.where(class_ids == class_id)[0]
        class_boxes = boxes[class_indices,:]
        class_scores = scores[class_indices]

        class_keep_boxes = nms(class_boxes, class_scores, iou_threshold)
        keep_boxes.extend(class_indices[class_keep_boxes])

    return keep_boxes

# 计算IoU
def compute_iou(box, boxes):
    # 计算两个框的xmin, ymin, xmax, ymax
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # 计算交集面积
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # 计算并集面积
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # 计算IoU
    iou = intersection_area / union_area

    return iou

# 将xywh格式转换为xyxy格式
def xywh2xyxy(x):
    # 将边界框 (x, y, w, h) 转换为边界框 (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

# 绘制检测结果
def draw_detections(image, boxes, scores, class_ids, mask_alpha=0.3):
    det_img = image.copy()

    img_height, img_width = image.shape[:2]
    font_size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)

    det_img = draw_masks(det_img, boxes, class_ids, mask_alpha)

    # 移除ONNX检测框绘制逻辑
    # for class_id, box, score in zip(class_ids, boxes, scores):
    #     color = colors[class_id]
    #     draw_box(det_img, box, color)
    #     label = class_names[class_id]
    #     caption = f'{label} {int(score * 100)}%'
    #     draw_text(det_img, caption, box, color, font_size, text_thickness)

    return det_img

# 绘制边界框
def draw_box(image, box, color=(0, 0, 255), thickness=2):
    x1, y1, x2, y2 = box.astype(int)
    return cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

# 绘制文本
def draw_text(image, text, box, color=(0, 0, 255), font_size=0.001, text_thickness=2):
    x1, y1, x2, y2 = box.astype(int)
    (tw, th), _ = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                  fontScale=font_size, thickness=text_thickness)
    th = int(th * 1.2)

    cv2.rectangle(image, (x1, y1),
                  (x1 + tw, y1 - th), color, -1)

    return cv2.putText(image, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), text_thickness, cv2.LINE_AA)

# 绘制掩码
def draw_masks(image, boxes, classes, mask_alpha=0.3):
    mask_img = image.copy()

    # 绘制边界框和标签
    for box, class_id in zip(boxes, classes):
        color = colors[class_id]

        x1, y1, x2, y2 = box.astype(int)

        # 在掩码图像中绘制填充矩形
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

    return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)

# 目标检测类
class RuneDetection:
    def __init__(self, model_path, conf_thres=0.5, iou_thres=0.3, gamma=0.3):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.gamma = gamma  # 添加gamma参数

        # 初始化模型
        self.initialize_model(model_path)

    def __call__(self, image):
        return self.detect_objects(image)

    def initialize_model(self, path):
        # 使用CPU进行推理
        self.session = onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider'])
        # 获取模型信息
        self.get_input_details()
        self.get_output_details()

    def detect_objects(self, image):
        input_tensor = self.prepare_input(image)

        # 对图像进行推理
        outputs = self.inference(input_tensor)

        self.boxes, self.scores, self.class_ids = self.process_output(outputs)

        return self.boxes, self.scores, self.class_ids

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 调整输入图像大小
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # 添加gamma校正调整曝光
        inv_gamma = 1.0 / self.gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        input_img = cv2.LUT(input_img, table)

        # 将输入像素值缩放到0到1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        print(f"推理时间: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs

    def process_output(self, output):
        predictions = np.squeeze(output[0]).T

        # 过滤掉低于阈值的对象置信度分数
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        # 获取具有最高置信度的类别
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # 获取每个对象的边界框
        boxes = self.extract_boxes(predictions)

        # 应用非极大值抑制以抑制弱的、重叠的边界框
        indices = multiclass_nms(boxes, scores, class_ids, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions):
        # 从预测中提取框
        boxes = predictions[:, :4]

        # 将框缩放到原始图像尺寸
        boxes = self.rescale_boxes(boxes)

        # 将框转换为xyxy格式
        boxes = xywh2xyxy(boxes)

        return boxes

    def rescale_boxes(self, boxes):
        # 将框重新缩放到原始图像尺寸
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):
        return draw_detections(image, self.boxes, self.scores, self.class_ids, mask_alpha)

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        # 根据用户提供的信息，模型大小为320*320
        self.input_height = 320
        self.input_width = 320

        print(f"模型输入尺寸: {self.input_width}x{self.input_height}")

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

# 主函数
def main():
    # 模型和视频路径
    model_path = "rune_detect_320.onnx"
    video_path = "rm_rune.mp4"
    output_path = "output_rune.mp4"
    
    # 初始化检测器
    detector = RuneDetection(model_path, conf_thres=0.5, iou_thres=0.3, gamma=0.5)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    centers = []  # 存储前五帧中心点
    avg_center = None  # 平均坐标
    frame_threshold = 5  # 前五帧
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        frame_start_time = time.perf_counter()  # 记录帧处理开始时间
        print(f"处理帧 {frame_count}")
        
        # 检测对象
        boxes, scores, class_ids = detector.detect_objects(frame)
        
        # 禁用原始检测框绘制
        result_frame = frame.copy()  # 使用原始帧替代检测结果帧

        # 收集前五帧类别1中心点
        if frame_count <= 5 and len(class_ids) > 0:
            # 只取第一个类别1的检测框
            first_class1_idx = next((i for i, cls_id in enumerate(class_ids) if cls_id == 1), None)
            if first_class1_idx is not None:
                box = boxes[first_class1_idx]
                center_x = int((box[0] + box[2])/2)
                center_y = int((box[1] + box[3])/2)
                centers.append((center_x, center_y))

        # 计算平均中心点
        if frame_count == 5 and len(centers) > 0:
            avg_center = np.mean(centers, axis=0).astype(int)

        # 绘制平均中心点
        if frame_count > 5 and avg_center is not None:
            cv2.circle(result_frame, tuple(avg_center), 5, (255, 0, 0), -1)

        # 应用gamma校正到显示帧
        gamma = detector.gamma
        inv_gamma = 1.0 / gamma
        lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        result_frame = cv2.LUT(result_frame, lut)

        # 处理类别0的检测框
        for idx, cls_id in enumerate(class_ids):
            if cls_id == 0:
                box = boxes[idx]
                x1, y1, x2, y2 = map(int, box)
                
                # 截取ROI区域
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                
                # 灰度化
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
       
                # 二值化
                _, binary_roi = cv2.threshold(gray_roi, 200, 255, cv2.THRESH_BINARY)

                # 查找轮廓（使用RETR_LIST获取全部轮廓）
                contours, _ = cv2.findContours(binary_roi, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                
                # 设置最小轮廓面积阈值（单位：像素）
                min_contour_area = 50
                
                # 过滤小面积轮廓
                filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
                
                # 动态过滤逻辑（强制保留两个最接近的候选框）
                history_areas = deque(maxlen=5)
                current_contour_areas = [cv2.contourArea(cnt) for cnt in filtered_contours]
                
                if len(history_areas) >= 3 and len(current_contour_areas) > 0:
                    avg_area = np.mean(history_areas)
                    std_dev = np.std(history_areas)
                    
                    # 筛选面积在3倍标准差范围内的候选
                    valid_indices = [i for i, a in enumerate(current_contour_areas)
                                    if abs(a - avg_area) < 3*std_dev]
                    
                    # 强制保留最接近的两个候选框
                    if len(valid_indices) >= 2:
                        sorted_indices = sorted(valid_indices, 
                                              key=lambda x: abs(current_contour_areas[x]-avg_area))[:2]
                        filtered_contours = [filtered_contours[i] for i in sorted_indices]
                        history_areas.extend([current_contour_areas[i] for i in sorted_indices])
                    else:
                        # 有效候选不足时沿用历史数据
                        filtered_contours = prev_valid_contours if len(prev_valid_contours)>=2 else []
                else:
                    # 初始化阶段保留面积最大的两个候选
                    if len(filtered_contours) >= 2:
                        sorted_indices = sorted(range(len(current_contour_areas)), 
                                              key=lambda i: current_contour_areas[i], reverse=True)[:2]
                        filtered_contours = [filtered_contours[i] for i in sorted_indices]
                        history_areas.extend([current_contour_areas[i] for i in sorted_indices])
                
                prev_valid_contours = filtered_contours.copy()
                
                # 绘制最小外接矩形
                for cnt in filtered_contours:
                    rect = cv2.minAreaRect(cnt)
                    box_pts = cv2.boxPoints(rect).astype(int)
                    # 转换到全局坐标系
                    global_box_pts = box_pts + np.array([x1, y1])
                    # 在主画面绘制绿色外接矩形
                    cv2.drawContours(result_frame, [global_box_pts.astype(int)], 0, (0,255,0), 2)
                    # 在ROI绘制原有矩形
                    cv2.drawContours(roi, [box_pts], 0, (0,255,0), 2)
                
                # 显示完整处理流程（原图+二值化+膨胀）
                processed_roi = cv2.cvtColor(binary_roi, cv2.COLOR_GRAY2BGR)
                combined = np.hstack((roi, processed_roi))
                cv2.imshow(f'Class 0 Processing #{idx}', combined)

        # 计算并输出帧处理总时间
        frame_end_time = time.perf_counter()
        frame_process_time = (frame_end_time - frame_start_time) * 1000
        print(f"帧 {frame_count} 总处理时间: {frame_process_time:.2f} ms")
        
        # 写入输出视频
        out.write(result_frame)
        
        # 显示结果
        cv2.imshow("Rune Detection", result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"处理完成，输出视频保存为: {output_path}")

if __name__ == "__main__":
    main()

from collections import deque

def main():
    # 初始化历史记录
    history_areas = deque(maxlen=5)
    prev_valid_contours = []
    
    # 在轮廓处理循环中添加过滤逻辑
    current_contour_areas = [cv2.contourArea(cnt) for cnt in filtered_contours]
    
    # 动态过滤逻辑（使用正确Python注释）
    if len(history_areas) >= 3:
        avg_area = np.mean(history_areas)
        std_dev = np.std(history_areas)
        
        # 计算面积差异并筛选（放宽至3倍标准差）
        valid_indices = [i for i, a in enumerate(current_contour_areas)
                        if abs(a - avg_area) < 3*std_dev]
        
        # 保留最接近的两个轮廓（增强稳定性）
        if len(valid_indices) >= 2:
            sorted_indices = sorted(valid_indices, 
                                  key=lambda x: abs(current_contour_areas[x]-avg_area))[:2]
            filtered_contours = [filtered_contours[i] for i in sorted_indices]
            history_areas.extend([current_contour_areas[i] for i in sorted_indices])
            prev_valid_contours = filtered_contours.copy()
        else:
            # 当有效轮廓不足时，智能沿用历史数据
            filtered_contours = prev_valid_contours if len(prev_valid_contours)>=2 else []
    else:
        # 初始化阶段保留面积最大的两个（增强鲁棒性）
        if len(filtered_contours) >= 2:
            sorted_indices = sorted(range(len(current_contour_areas)), 
                                  key=lambda i: current_contour_areas[i], reverse=True)[:2]
            filtered_contours = [filtered_contours[i] for i in sorted_indices]
            history_areas.extend([current_contour_areas[i] for i in sorted_indices])