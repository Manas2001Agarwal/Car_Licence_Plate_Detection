from ultralytics import YOLO

def model_train():
    
    model = YOLO('yolov8n.pt')
    
    model.train(
            data='datasets/datasets.yaml',  # Path to the dataset configuration file
            epochs=100,                     # Number of training epochs
            batch=8,                       # Batch size
            device='mps',                   # Use GPU for training
            imgsz=320,                      # Image size (width and height) for training
            cache=True
        )
    
if __name__ == "__main__":
    model_train()
    
    