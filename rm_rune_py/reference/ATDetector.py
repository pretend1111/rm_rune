import cv2
from detection.target_detection import TargetDetection
from detection.utils import draw_detections

# yolov8 onnx 模型推理
class ATDetector():
    def __init__(self):
        super(ATDetector, self).__init__()
        self.model_path = "../yolov8s_best.onnx"
        self.detector = TargetDetection(self.model_path, conf_thres=0.5, iou_thres=0.3)

    def detect_image(self, input_image, output_image):
        cv_img = cv2.imread(input_image)
        boxes, scores, class_ids = self.detector.detect_objects(cv_img)
        cv_img = draw_detections(cv_img, boxes, scores, class_ids)
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        cv2.imwrite(output_image, cv_img)
        cv2.imshow('output', cv_img)
        cv2.waitKey(0)

    def detect_video(self, input_video, output_video):
        cap = cv2.VideoCapture(input_video)
        fps = int(cap.get(5))
        videoWriter = None

        while True:
            _, cv_img = cap.read()
            if cv_img is None:
                break
            boxes, scores, class_ids = self.detector.detect_objects(cv_img)
            cv_img = draw_detections(cv_img, boxes, scores, class_ids)

            # 如果视频写入器未初始化，则使用输出视频路径和参数进行初始化
            if videoWriter is None:
                fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                # 在这里给值了，它就不是None, 下次判断它就不进这里了
                videoWriter = cv2.VideoWriter(output_video, fourcc, fps, (cv_img.shape[1], cv_img.shape[0]))

            videoWriter.write(cv_img)
            cv2.imshow("aod", cv_img)
            cv2.waitKey(5)

            # 等待按键并检查窗口是否关闭
            if cv2.getWindowProperty("aod", cv2.WND_PROP_AUTOSIZE) < 1:
                # 点x退出
                break

        cap.release()
        videoWriter.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    det = ATDetector()
    # input_image = "../data/A_905.jpg"
    # output_image = '../data/output.jpg'
    # det.detect_image(input_image, output_image)
    input_video=r"E:\dataset\MOT\video\A13.mp4"
    output_video="../data/output.mp4"
    det.detect_video(input_video,output_video)

