import os
import torch
import numpy as np
import cv2
from ultralytics import YOLO
from collections import deque
import supervision as sv
from typing import Dict, List, Optional, Tuple, Union

class ObjectDetector:
    """
    Object detection using YOLOv11 from Ultralytics with ByteTracker integration
    """
    def __init__(self, model_size='small', model_path=None, conf_thres=0.25, iou_thres=0.45, 
                 classes=None, device=None, image_size=None, tracking_config=None):
        """
        Initialize the object detector
        
        Args:
            model_size (str): Model size ('nano', 'small', 'medium', 'large', 'extra')
            model_path (str): Direct path to a .pt model file (takes precedence over model_size)
            conf_thres (float): Confidence threshold for detections
            iou_thres (float): IoU threshold for NMS
            classes (list): List of classes to detect (None for all classes)
            device (str): Device to run inference on ('cuda', 'cpu', 'mps')
            image_size (list|tuple|int): Image size for processing [width, height] or single int value
            tracking_config (dict): Configuration for ByteTracker
        """
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        self.device = device
        
        # Set MPS fallback for operations not supported on Apple Silicon
        if self.device == 'mps':
            print("Using MPS device with CPU fallback for unsupported operations")
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        
        print(f"Using device: {self.device} for object detection")
        
        # Set image size if provided
        self.image_size = image_size
        if image_size:
            print(f"Using image size: {image_size} for object detection")
        
        # Map model size to model name
        model_map = {
            'nano': 'yolo11n',
            'small': 'yolo11s',
            'medium': 'yolo11m',
            'large': 'yolo11l',
            'extra': 'yolo11x'
        }
        
        # Load model
        try:
            if model_path and os.path.exists(model_path):
                self.model = YOLO(model_path)
                print(f"Loaded YOLOv11 model from {model_path} on {self.device}")
            else:
                if model_path:
                    print(f"Warning: Model path {model_path} not found, falling back to default model")
                model_name = model_map.get(model_size.lower(), model_map['small'])
                self.model = YOLO(model_name)
                print(f"Loaded YOLOv11 {model_size} model on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Trying to load with default settings...")
            model_name = model_map.get(model_size.lower(), model_map['small'])
            self.model = YOLO(model_name)
        
        # Set model parameters
        self.model.overrides['conf'] = conf_thres
        self.model.overrides['iou'] = iou_thres
        self.model.overrides['agnostic_nms'] = False
        self.model.overrides['max_det'] = 1000
        
        if classes is not None:
            self.model.overrides['classes'] = classes
        
        # Initialize tracking if config provided
        self.tracking_config = tracking_config if tracking_config else {}
        
        # Initialize ByteTracker configuration
        self.tracker = sv.ByteTrack(
            lost_track_buffer=self.tracking_config.get('lost_track_buffer', 0.25),
            track_activation_threshold=self.tracking_config.get('track_activation_threshold', 30),
            minimum_matching_threshold=self.tracking_config.get('minimum_matching_threshold', 0.8),
        )

        
        # Initialize detection smoothing if configured
        filter_config = self.tracking_config.get('filter_config', {})
        length = filter_config.get('length', 0.2)
        
        # Initialize box smoothing
        self.smoother = sv.DetectionsSmoother(length=length)
        
        # Initialize tracking trajectories for visualization
        self.tracking_trajectories = {}
    
    def detect(self, image, track=True):
        """
        Detect objects in an image
        
        Args:
            image (numpy.ndarray): Input image (BGR format)
            track (bool): Whether to track objects across frames
            
        Returns:
            tuple: (annotated_image, detections)
                - annotated_image (numpy.ndarray): Image with detections drawn
                - detections (list): List of detections [bbox, score, class_id, object_id]
        """
        # Make a copy of the image for annotation
        annotated_image = image.copy()
        
        try:
            # Run inference with specified image size if provided
            if track:
                kwargs = {"verbose": False, "device": self.device, "persist": True}
                if self.image_size is not None:
                    kwargs["imgsz"] = self.image_size
                results = self.model.track(image, **kwargs)
            else:
                kwargs = {"verbose": False, "device": self.device}
                if self.image_size is not None:
                    kwargs["imgsz"] = self.image_size
                results = self.model.predict(image, **kwargs)
        except RuntimeError as e:
            # Handle potential MPS errors
            if self.device == 'mps' and "not currently implemented for the MPS device" in str(e):
                print(f"MPS error during detection: {e}")
                print("Falling back to CPU for this frame")
                if track:
                    kwargs = {"verbose": False, "device": 'cpu', "persist": True}
                    if self.image_size is not None:
                        kwargs["imgsz"] = self.image_size
                    results = self.model.track(image, **kwargs)
                else:
                    kwargs = {"verbose": False, "device": 'cpu'}
                    if self.image_size is not None:
                        kwargs["imgsz"] = self.image_size
                    results = self.model.predict(image, **kwargs)
            else:
                # Re-raise the error if not MPS or not an implementation error
                raise
        
        detections = []
        
        # Extract detections from YOLO results
        yolo_detections = self._extract_detections_from_results(results)
        
        if track and yolo_detections is not None:
            # Use ByteTracker for tracking
            frame_height, frame_width = image.shape[:2]
            
            # Convert detections to supervision format
            sv_detections = sv.Detections(
                xyxy=yolo_detections['bboxes'],
                confidence=yolo_detections['scores'],
                class_id=yolo_detections['class_ids']
            )
            
            # Apply tracking
            sv_detections = self.tracker.update_with_detections(sv_detections)
            
            # Apply smoothing to tracked objects
            sv_detections = self.smoother.update_with_detections(sv_detections)
            
            # Convert back to our format
            for i in range(len(sv_detections.xyxy)):
                bbox = sv_detections.xyxy[i].tolist()
                score = float(sv_detections.confidence[i]) if sv_detections.confidence is not None else 1.0
                class_id = int(sv_detections.class_id[i]) if sv_detections.class_id is not None else 0
                track_id = int(sv_detections.tracker_id[i]) if sv_detections.tracker_id is not None else None
                
                detections.append([
                    bbox,              # bbox [x1, y1, x2, y2]
                    score,             # confidence score
                    class_id,          # class id
                    track_id           # track id
                ])
                
                # Update trajectories for visualization
                if track_id is not None:
                    centroid_x = (bbox[0] + bbox[2]) / 2
                    centroid_y = (bbox[1] + bbox[3]) / 2
                    
                    if track_id not in self.tracking_trajectories:
                        self.tracking_trajectories[track_id] = deque(maxlen=10)
                    
                    self.tracking_trajectories[track_id].append((centroid_x, centroid_y))
            
            # Annotate image with detections
            self._annotate_image(annotated_image, detections, results)
            
            # Draw trajectories
            self._draw_trajectories(annotated_image)
        else:
            # For non-tracking mode, just extract detections
            if yolo_detections is not None:
                for i in range(len(yolo_detections['bboxes'])):
                    detections.append([
                        yolo_detections['bboxes'][i].tolist(),
                        float(yolo_detections['scores'][i]),
                        int(yolo_detections['class_ids'][i]),
                        None  # No tracking ID
                    ])
                    
                # Annotate image with detections
                self._annotate_image(annotated_image, detections, results)
        
        return annotated_image, detections
    
    def _extract_detections_from_results(self, results):
        """
        Extract detections from YOLO results
        
        Args:
            results: YOLO results
            
        Returns:
            dict: Dictionary with bboxes, scores, class_ids
        """
        all_bboxes = []
        all_scores = []
        all_class_ids = []
        
        for predictions in results:
            if predictions is None or predictions.boxes is None:
                continue
                
            # Process boxes
            boxes = predictions.boxes
            
            if len(boxes) == 0:
                continue
                
            # Extract information
            bboxes = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy()
            
            all_bboxes.extend(bboxes)
            all_scores.extend(scores)
            all_class_ids.extend(class_ids)
        
        if not all_bboxes:
            return None
            
        return {
            'bboxes': np.array(all_bboxes),
            'scores': np.array(all_scores),
            'class_ids': np.array(all_class_ids)
        }
    
    def _annotate_image(self, image, detections, results):
        """
        Annotate image with detections
        
        Args:
            image (numpy.ndarray): Image to annotate
            detections (list): List of detections
            results: YOLO results for class names
        """
        for detection in detections:
            bbox, score, class_id, obj_id = detection
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Get class name
            class_name = self.get_class_name(class_id, results)
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 225), 2)
            
            # Add label
            if obj_id is not None:
                label = f"ID: {obj_id} {class_name} {score:.2f}"
            else:
                label = f"{class_name} {score:.2f}"
                
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            dim, baseline = text_size[0], text_size[1]
            cv2.rectangle(image, (x1, y1), (x1 + dim[0], y1 - dim[1] - baseline), (30, 30, 30), cv2.FILLED)
            cv2.putText(image, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _draw_trajectories(self, image):
        """
        Draw trajectories for tracked objects
        
        Args:
            image (numpy.ndarray): Image to draw on
        """
        for id_, trajectory in self.tracking_trajectories.items():
            for i in range(1, len(trajectory)):
                thickness = int(2 * (i / len(trajectory)) + 1)
                cv2.line(image, 
                        (int(trajectory[i-1][0]), int(trajectory[i-1][1])), 
                        (int(trajectory[i][0]), int(trajectory[i][1])), 
                        (255, 255, 255), thickness)
    
    def get_class_name(self, class_id, results):
        """
        Get class name from class ID
        
        Args:
            class_id (int): Class ID
            results: YOLO results containing names dictionary
            
        Returns:
            str: Class name
        """
        for predictions in results:
            if predictions is not None and hasattr(predictions, 'names'):
                return predictions.names.get(int(class_id), f"Class {class_id}")
        
        # Fallback to model names
        return self.model.names.get(int(class_id), f"Class {class_id}")
    
    def get_class_names(self):
        """
        Get the names of the classes that the model can detect
        
        Returns:
            dict: Dictionary mapping class IDs to class names
        """
        return self.model.names