import cv2
import torch
import numpy as np
import logging
import pathlib

pathlib.PosixPath = pathlib.WindowsPath

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def attempt_load_custom(weights, device='cpu'):
    """Custom function to load YOLOv5 model with weights_only=False"""
    try:
        model = torch.load(weights, map_location=device, weights_only=False)
        if isinstance(model, dict):
            model = model['model']  
        
        model = model.float()
        model.eval()
        logger.info("YOLOv5 model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def main():
    model_path = "best.pt"

    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device in use: {device}")

    
    logger.info(f"PyTorch version: {torch.__version__}")
    try:
        import yolov5
        logger.info(f"YOLOv5 version: {yolov5.__version__}")
    except AttributeError:
        logger.warning("YOLOv5 version unknown. Manual installation may be used.")
    try:
        logger.info(f"OpenCV version: {cv2.__version__}")
    except Exception as e:
        logger.error(f"Error checking OpenCV version: {e}")
        return

    try:
        model = attempt_load_custom(model_path, device=device)
    except Exception as e:
        logger.error("Program stopped due to model loading error.")
        return

    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Error: Webcam could not be opened. Ensure webcam is connected.")
            return
    except Exception as e:
        logger.error(f"Error opening webcam: {e}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Error: Frame not received.")
                break

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img).to(device)
            img = img.permute(2, 0, 1).float() / 255.0  
            img = img.unsqueeze(0)  

            try:
                model.eval()
                with torch.no_grad():
                    pred = model(img)[0]
                from yolov5.utils.general import non_max_suppression
                pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.45)
            except Exception as e:
                logger.error(f"Error during detection: {e}")
                continue

            for det in pred:
                if len(det):
                    for *xyxy, conf, cls in det:
                        x1, y1, x2, y2 = map(int, xyxy)
                        confidence = float(conf)
                        class_id = int(cls)
                        class_name = model.names[class_id] if hasattr(model, 'names') else f"Class {class_id}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{class_name}: {confidence:.2f}"
                        cv2.putText(
                            frame,
                            label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )

            try:
                cv2.imshow("Hand Gesture Detection", frame)
            except Exception as e:
                logger.error(f"Error displaying image: {e}")
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Exiting program.")
                break

    except KeyboardInterrupt:
        logger.info("Program interrupted by user.")
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            logger.warning(f"Error closing windows: {e}")
        logger.info("Resources released.")

if __name__ == "__main__":
    main()