import os
import torch
import numpy as np
import cv2
from ultralytics import SAM
import supervision as sv

class SegmentationModel:
    """
    Object segmentation using SAM (Segment Anything Model) from Ultralytics
    """
    def __init__(self, model_size='base', model_path=None, device=None):
        """
        Initialize the segmentation model
        
        Args:
            model_size (str): Model size ('base', 'large')
            model_path (str): Direct path to a custom SAM model file (takes precedence over model_size)
            device (str): Device to run inference on ('cuda', 'cpu', 'mps')
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
        
        print(f"Using device: {self.device} for segmentation")
        
        # Map model size to model name
        model_map = {
            'base': 'sam_b.pt',
            'large': 'sam_l.pt'
        }
        
        # Determine which model to use
        if model_path and os.path.exists(model_path):
            model_name = model_path
            print(f"Using custom SAM model from: {model_path}")
        else:
            if model_path:
                print(f"Warning: Model path {model_path} not found, falling back to default model")
            model_name = model_map.get(model_size.lower(), model_map['base'])
            print(f"Using standard SAM {model_size} model")
        
        # Load model
        try:
            self.model = SAM(model_name)
            print(f"Loaded SAM model on {self.device}")
        except Exception as e:
            print(f"Error loading SAM model: {e}")
            print("Trying to load with default settings...")
            self.model = SAM('sam_b.pt')
    
        # Initialize color palette for visualization
        self.annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
        
        # Generate a random color map for consistent visualization
        self.color_map = {}
    
    def segment_bbox(self, image, bbox, class_id, class_name=None):
        """
        Segment an object within a bounding box using SAM
        
        Args:
            image (numpy.ndarray): Input image (BGR format)
            bbox (list): Bounding box [x1, y1, x2, y2]
            class_id (int): Class ID for color mapping
            class_name (str): Class name (optional)
            
        Returns:
            tuple: (segmentation_mask, colored_mask)
                - segmentation_mask (numpy.ndarray): Binary segmentation mask
                - colored_mask (numpy.ndarray): Colored mask for visualization
        """
        # Ensure bbox coordinates are integers
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Calculate center point (foreground prompt)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        try:
            # Extract region of interest to reduce computation
            roi = image[max(0, y1-10):min(image.shape[0], y2+10), 
                         max(0, x1-10):min(image.shape[1], x2+10)]
            
            # Skip tiny ROIs
            if roi.shape[0] < 10 or roi.shape[1] < 10:
                return None, image
            
            # Use point prompts method directly
            results = self.model.predict(
                source=roi,  # Use ROI instead of full image
                device=self.device,
                retina_masks=True,
                imgsz=1024,
                conf=0.25,
                iou=0.9,
                # Important: don't use points or bboxes here - SAM can auto-segment small ROIs
            )
            
            # Extract mask from results
            if results and hasattr(results[0], 'masks') and results[0].masks is not None and len(results[0].masks.data) > 0:
                # Get the mask - take the first one if multiple are returned
                mask_roi = results[0].masks.data[0].cpu().numpy()
                
                # Create full-sized mask
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                
                # Place ROI mask in the correct position
                roi_y1, roi_y2 = max(0, y1-10), min(image.shape[0], y2+10)
                roi_x1, roi_x2 = max(0, x1-10), min(image.shape[1], x2+10)
                
                # Resize mask_roi to match ROI dimensions if needed
                if mask_roi.shape[:2] != (roi_y2 - roi_y1, roi_x2 - roi_x1):
                    mask_roi = cv2.resize(mask_roi.astype('uint8'), 
                                          (roi_x2 - roi_x1, roi_y2 - roi_y1))
                
                # Place mask in the correct position
                mask[roi_y1:roi_y2, roi_x1:roi_x2] = mask_roi
                
                # Determine color based on class
                if class_name:
                    if class_name.lower() not in self.color_map:
                        if 'car' in class_name.lower() or 'vehicle' in class_name.lower():
                            color = (0, 0, 255)  # Red for vehicles
                        elif 'person' in class_name.lower():
                            color = (0, 255, 0)  # Green for persons
                        elif 'bicycle' in class_name.lower() or 'motorcycle' in class_name.lower():
                            color = (255, 0, 0)  # Blue for bikes
                        elif 'ball' in class_name.lower() or 'sports ball' in class_name.lower():
                            color = (0, 165, 255)  # Orange for balls
                        else:
                            # Generate a random color for other classes
                            color = tuple(map(int, np.random.randint(100, 255, size=3)))
                        
                        self.color_map[class_name.lower()] = color
                    else:
                        color = self.color_map[class_name.lower()]
                else:
                    # Use class_id to make consistent colors
                    if class_id not in self.color_map:
                        color = tuple(map(int, np.random.randint(100, 255, size=3)))
                        self.color_map[class_id] = color
                    else:
                        color = self.color_map[class_id]
                
                # Create colored overlay
                colored_image = self.create_overlay(image, mask, color, alpha=0.5)
                
                return mask, colored_image
            else:
                print(f"No valid mask found for {class_name if class_name else f'class {class_id}'}")
                return None, image
                
        except Exception as e:
            print(f"Error during segmentation: {e}")
            # Try an alternative method with automatic segmentation
            try:
                # Just use the bounding box as a prompt
                results = self.model.predict(
                    source=image,
                    device=self.device,
                    bboxes=np.array([[x1, y1, x2, y2]]),
                    retina_masks=True,
                    imgsz=1024,
                    conf=0.25,
                    iou=0.9
                )
                
                if results and hasattr(results[0], 'masks') and results[0].masks is not None and len(results[0].masks.data) > 0:
                    mask = results[0].masks.data[0].cpu().numpy()
                    
                    # Resize mask to original image size if needed
                    if mask.shape[:2] != image.shape[:2]:
                        mask = cv2.resize(mask.astype('uint8'), (image.shape[1], image.shape[0]))
                    
                    # Get or create color
                    if class_name and class_name.lower() in self.color_map:
                        color = self.color_map[class_name.lower()]
                    elif class_id in self.color_map:
                        color = self.color_map[class_id]
                    else:
                        # Default color
                        color = (0, 255, 255)
                    
                    # Create colored overlay
                    colored_image = self.create_overlay(image, mask, color, alpha=0.5)
                    
                    return mask, colored_image
                else:
                    return None, image
            except:
                # Fall back to a simple box-shaped mask if SAM fails
                print(f"Falling back to simple box-shaped mask for {class_name if class_name else f'class {class_id}'}")
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 1, -1)  # -1 means filled rectangle
                
                if class_name and class_name.lower() in self.color_map:
                    color = self.color_map[class_name.lower()]
                elif class_id in self.color_map:
                    color = self.color_map[class_id]
                else:
                    # Default color
                    color = (0, 255, 255)
                
                colored_image = self.create_overlay(image, mask, color, alpha=0.3)
                return mask, colored_image
    
    def create_overlay(self, image, mask, color, alpha=0.5):
        """
        Create a colored overlay for the segmentation mask
        
        Args:
            image (numpy.ndarray): Original image
            mask (numpy.ndarray): Binary segmentation mask
            color (tuple): Color in BGR format
            alpha (float): Transparency factor (0-1)
            
        Returns:
            numpy.ndarray: Image with colored overlay
        """
        if mask is None:
            return image
        
        # Ensure mask is binary
        if mask.max() > 1:
            mask = mask.astype(bool).astype(np.uint8)
        
        # Create colored mask
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = color
        
        # Create overlay
        overlay = image.copy()
        cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0, overlay)
        
        return overlay