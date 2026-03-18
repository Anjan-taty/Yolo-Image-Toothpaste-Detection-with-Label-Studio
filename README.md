# 🧠 YOLO Custom Object Detection

This project performs **real-time object detection** using a custom trained **YOLO model** on webcam, images, or videos.

The model is trained using the **Ultralytics YOLO framework** and runs inference through the `main.py` script.

---

## 📂 Project Structure

```
YOLO/
│
├── data/
│   └── train/
│       └── my_model.pt        # 🎯 Trained YOLO model
│
├── main.py                    # 🚀 Main detection script
├── data_initial.zip           # 📦 Initial dataset
├── data.zip                   # 📦 Processed dataset
└── README.md                  # 📘 Documentation
```

---

## ⚙️ Requirements

Make sure you have the following installed:

- 🐍 Python 3.9+
- ⚡ CUDA-supported GPU (optional but recommended)

---

## 🛠️ Installation

### 1. Clone the Repository

```
git clone https://github.com/Anjan-taty/Yolo-Image-Toothpaste-Detection-with-Label-Studio.git
cd Yolo-Image-Toothpaste-Detection-with-Label-Studio
```

Or download the ZIP and extract it.

---

### 2. Create Virtual Environment

```
python -m venv myenv
```

Activate environment:

**🪟 Windows**
```
myenv\Scripts\activate
```

**🐧 Linux / 🍎 Mac**
```
source myenv/bin/activate
```

---

### 3. Install Dependencies

```
pip install -r requirements.txt
```

### 🔥 (Optional) Install PyTorch with CUDA

```
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

---

## 🎯 Running the Detection

Run the detection script:

```
python main.py --model data/my_model.pt --source usb0 --resolution 1280x720
```

---

## ⚙️ Parameters

| Parameter      | Description                   |
| -------------- | ----------------------------- |
| `--model`      | Path to trained YOLO model    |
| `--source`     | Image / Video / Webcam source |
| `--resolution` | Output display resolution     |
| `--thresh`     | Confidence threshold          |

### Example:

```
python main.py --model data/my_model.pt --source usb0 --resolution 1280x720 --thresh 0.8
```

---

## 📸 Source Options

| Source  | Example     |
|--------|------------|
| 🎥 Webcam | `usb0`      |
| 🖼️ Image  | `test.jpg`  |
| 🎬 Video  | `video.mp4` |
| 📁 Folder | `images/`   |

Example:

```
python main.py --model data/my_model.pt --source test.jpg
```

---

## 📊 Output

The program displays:

- 📦 Bounding boxes  
- 🏷️ Object class names  
- 📈 Confidence scores  
- ⚡ Real-time FPS  
- 🔢 Object count  

Press **Q** to quit the program.

---

## 📦 Dependencies

Main libraries used:

```
ultralytics
torch
torchvision
torchaudio
opencv-python
numpy
```

Install manually if needed:

```
pip install ultralytics opencv-python numpy
```

---

## 📝 Notes

- Ensure the model file `my_model.pt` exists in the `data/` directory.
- Webcam index may vary depending on the system.

Example:

```
usb0
usb1
```

---

## 🎯 Use Case

This project is designed for:

- 🔍 Custom object detection  
- 🎥 Real-time video inference  
- 🧠 Computer vision experimentation  
- 🚀 Deployable AI applications  

---

## 👨‍💻 Author

Developed by **Anjan (Taty)**  
Passionate about AI, Computer Vision, and real-world applications 🚀

---

## ⭐ Support

If you like this project, consider giving it a ⭐ on GitHub!
