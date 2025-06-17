# 🎵 Sequence Generation Model  (LyricGenAI)

This repository contains the code and resources for training and generating sequences using a deep learning model.  

---

## 🚀 model architecture 

![{A9212516-5CBE-4370-9A46-36A66691C7FA}](https://github.com/user-attachments/assets/7e4b343b-39c4-4248-980f-3320a94f5dcd)



## 📂 Repository Structure  
- **`model.py`** → Code for training the model.  
- **`dataset`** → Contains the dataset link used for training.  
- **`model_paras`** → 🤗 link to the trained model (`model.pth`).  
- **`play.ipynb`** → Jupyter Notebook to load the trained model and generate sequences.  
---

## 🚀 Getting Started  

1️⃣ Clone the Repository  

    git clone https://github.com/FRIDAYFRINGE/LyricGenAI.git
    cd LyricGenAI

2️⃣ Download the Trained Model
Since GitHub has file size limits, download the model (model.pth) from 🤗:[Download model.pth](https://huggingface.co/spaces/fridayfringe/lgai/tree/main)

    After downloading, place model.pth in the same directory as play.ipynb.

3️⃣ Install Dependencies

    !pip install -r requirements.txt

4️⃣ Train the Model
To train the model from scratch, run:

    python model.py


5️⃣ Run Inference
Open the Jupyter Notebook:
    
    jupyter notebook play.ipynb



🧠 Model Details
Architecture: Transformer-based GPT model
Total Parameters: 19,446,528 (20M)
Embedding Size: 256
Number of Layers: 8
Number of Heads: 8
Max Sequence Length: 1024



🔧 Example: Loading the Model-
      
    # Load Weights
    model.load_state_dict(torch.load("model.pth", map_location=torch.device('gpu')))
    model.eval()



📌 Dataset
The dataset used for training can be found here:
📂 [Dataset Link](https://www.kaggle.com/datasets/vatsalmavani/spotify-dataset)

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.






