# Fruit Image Classification

## Overview
This project implements a deep learning model for classifying images of various fruits including apple, banana, grape, cantaloupe, strawberries, orange, mango, and more. The model is built using MobileNetV2 architecture with transfer learning to efficiently classify fruits with high accuracy.

## Dataset
- **Total Images**: ~31,000 images
- **Classes**: 11 fruit categories
- **Images per Category**: ~3,000 images per fruit
- **Sources**: 
  - Unsplash
  - Pexels
  - Pixabay
  - Fruits-262 dataset from Kaggle
  - Other freely available fruit datasets from Kaggle

## Project Motivation
The primary goal of this project is to develop an accurate and lightweight fruit classification system that can run efficiently on devices with limited computational resources. This has several practical applications:

1. **Mobile Applications**: Enabling users to identify fruits using their smartphone cameras
2. **Agricultural Technology**: Assisting in fruit sorting and quality control systems
3. **Educational Tools**: Creating interactive learning applications for botany and nutrition
4. **Smart Kitchen Devices**: Integration with smart refrigerators and cooking assistants

MobileNetV2 was specifically chosen as the backbone architecture because:
- **Efficiency**: Designed to run on resource-constrained devices (like our Intel i3 system)
- **Small Size**: Requires less memory while maintaining high accuracy
- **Transfer Learning**: Leverages pre-trained weights on ImageNet to reduce training time
- **Lightweight**: Optimized for mobile and embedded vision applications

## Project Structure
```
fruit-classification/
│
├── train_model.py           # Initial training script
├── resume_training.py       # Script to resume training from checkpoints
├── checkpoints/             # Directory containing model checkpoints
│   ├── model_epoch_XX_val_acc_XX.h5   # Checkpoint files
│   └── final_model.h5       # Final trained model
│
├── data/                    # Dataset directory
│   └── Plants - Copy/       # Directory containing fruit images
│       ├── apple/           # Subdirectories for each fruit category
│       ├── banana/
│       ├── grape/
│       └── ...
│
└── README.md                # This file
```

## Technical Implementation
- **Framework**: TensorFlow/Keras
- **Base Model**: MobileNetV2 pre-trained on ImageNet
- **Image Size**: 224×224 pixels (standard for MobileNetV2)
- **Training Strategy**: 
  - Transfer learning with frozen base model
  - Custom classification head
  - Data augmentation for improved generalization
  - Checkpoint saving for resilient training
  - Early stopping to prevent overfitting

## Installation and Setup
1. Clone the repository
2. Install dependencies:
   ```
   pip install tensorflow numpy pillow
   ```
3. Organize your fruit images in subdirectories as described in the project structure
4. Update the data path in the scripts to point to your dataset location

## Usage
### Initial Training
```
python train_model.py
```

### Resuming Training from Checkpoint
```
python resume_training.py
```

## Model Performance
The model achieves high accuracy in fruit classification by leveraging transfer learning from MobileNetV2's pre-trained weights and applying custom data augmentation techniques to prevent overfitting.

## Future Improvements
- Implement model quantization for further size reduction
- Add real-time classification via webcam
- Create a simple web/mobile interface for easy testing
- Expand the dataset to include more fruit varieties and conditions

## License
This project uses images from various free stock image websites and publicly available datasets. All code in this repository is available for educational and research purposes.

## Acknowledgements
- The creators and contributors of the original datasets
- The TensorFlow and Keras teams
- The authors of MobileNetV2 architecture