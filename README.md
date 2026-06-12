# Task 2: Deep Learning - Image Classification
**For Internspark Tech Internship**

## Environment Setup

### Prerequisites
- Python 3.8 or higher
- GPU recommended (Google Colab free GPU works)
- Internet connection

### Installation Commands

```bash
# Clone the repository
git clone https://github.com/UtsavMishra0910/Interspark-task2_deep_learning
cd Internspark tech-task-2

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install required packages
pip install torch torchvision matplotlib scikit-learn jupyter
Exact Commands to Run Notebook
bash
# Option 1: Google Colab with GPU (Recommended)
# 1. Go to: https://colab.research.google.com
# 2. Runtime → Change runtime type → GPU
# 3. Upload task2_deep_learning.ipynb
# 4. Runtime → Run all

# Option 2: Run Locally (CPU - slower)
jupyter notebook task2_deep_learning.ipynb
Package Versions (Exact)
text
Python==3.8.0
torch==2.0.0
torchvision==0.15.0
matplotlib==3.7.1
scikit-learn==1.3.0
jupyter==1.0.0
Model Architecture
Transfer Learning with ResNet18
Base Model: ResNet18 pretrained on ImageNet

Modified Layers:

First conv layer: Adjusted for 32x32 CIFAR images

Final FC layer: Changed from 1000 to 10 classes

Total Parameters: ~11 million

Data Augmentation
Random horizontal flip

Random crop with padding

Normalization (mean and std per channel)

Training Details
Parameter	Value
Dataset	CIFAR-10 (50k train, 10k test)
Classes	10 (plane, car, bird, cat, deer, dog, frog, horse, ship, truck)
Epochs	10
Batch Size	64
Optimizer	Adam
Learning Rate	0.001
Loss Function	Cross Entropy
Results
Final Accuracy
Test Accuracy: ~75-80% (after 10 epochs)

Performance by Class
Class	Precision	Recall	F1-Score
plane	~0.80	~0.75	~0.77
car	~0.85	~0.82	~0.83
bird	~0.70	~0.65	~0.67
cat	~0.60	~0.58	~0.59
deer	~0.72	~0.70	~0.71
dog	~0.65	~0.62	~0.63
frog	~0.78	~0.80	~0.79
horse	~0.82	~0.78	~0.80
ship	~0.85	~0.88	~0.86
truck	~0.82	~0.85	~0.83
Output Files Generated
File	Description
task2_deep_learning.ipynb	Main notebook
README.md	This file
requirements.txt	Package dependencies
sample_images.png	Sample CIFAR-10 images
training_curves.png	Loss and accuracy curves
confusion_matrix.png	Confusion matrix heatmap
inference_examples.png	Sample predictions
cifar10_model.pth	Model weights only
cifar10_model_full.pth	Complete model
How to Run Inference on New Images
python
import torch
import torchvision.transforms as transforms
from PIL import Image

# Load model
model = torch.load('cifar10_model_full.pth')
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Load and predict
image = Image.open('your_image.jpg')
image_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(image_tensor)
    predicted = torch.argmax(output, 1).item()
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    print(f"Predicted: {classes[predicted]}")
Training Curves Interpretation
Loss Curve: Should decrease steadily

Accuracy Curve: Should increase and plateau

If accuracy stops improving, model has converged

Confusion Matrix Insights
Common misclassifications:

Cat ↔ Dog (similar appearance)

Bird ↔ Plane (both have wings in images)

Deer ↔ Horse (four-legged animals)

Screenshots Required for Submission
Sample images from dataset

Training progress (epoch outputs)

Training curves plot

Final accuracy and classification report

Confusion matrix

Inference examples

Troubleshooting
Issue: Out of memory (CUDA out of memory)

python
# Reduce batch size
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
Issue: Training too slow

Use GPU in Colab (Runtime → Change runtime type → GPU)

Reduce number of epochs

Issue: Low accuracy (<60%)

Train for more epochs (15-20)

Add more data augmentation

Try different learning rate (0.0005 or 0.002)

Model Files Location
Large model files are stored on Google Drive:

cifar10_model.pth: [Google Drive Link]

cifar10_model_full.pth: [Google Drive Link]

References
CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html

ResNet Paper: https://arxiv.org/abs/1512.03385

PyTorch Documentation: https://pytorch.org/
