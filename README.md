🛡️ Memory-Based Malware Detection Using Machine Learning and Deep Learning

A cybersecurity project that detects malicious software using memory-based behavioral features extracted from the CIC-MalMem-2022 dataset. The system combines Machine Learning (Random Forest) and Deep Learning (Multi-Layer Perceptron) to classify malware and benign processes based on runtime memory characteristics.

⸻

📌 Overview

Traditional malware detection techniques often struggle against modern threats that use obfuscation, encryption, and packing techniques. Instead of relying on static file analysis, this project analyzes memory-derived behavioral features, making detection more resilient against malware evasion techniques.

The project implements and compares:

* Multi-Layer Perceptron (MLP) – Deep Learning
* Random Forest – Machine Learning
* Streamlit Dashboard – Interactive visualization and model comparison

⸻

🚀 Features

* Memory-based malware classification
* Detection of malware and benign samples
* Comparison of ML and DL models
* Accuracy and ROC curve evaluation
* Confidence score generation
* Interactive Streamlit dashboard
* End-to-end preprocessing, training, and inference pipeline

⸻

📂 Dataset

CIC-MalMem-2022

The project uses the CIC-MalMem-2022 dataset developed by the Canadian Institute for Cybersecurity.

Dataset Characteristics:

* ~58,000 samples
* Malware and benign processes
* Memory-based behavioral features
* Suitable for detecting obfuscated malware

Official Dataset:

CIC-MalMem-2022 Dataset￼

⸻

⚙️ Preprocessing Pipeline

1. Remove metadata columns (Category, Filename)
2. Select numerical behavioral features
3. Normalize features using StandardScaler
4. Encode labels using LabelEncoder
5. Perform 80:20 train-test split

⸻

🧠 Models Used

Multi-Layer Perceptron (MLP)

* Built using PyTorch
* Fully connected neural network
* ReLU activation
* Adam optimizer
* Cross-Entropy Loss

Random Forest

* Implemented using Scikit-learn
* Ensemble of decision trees
* Majority voting classification
* High interpretability and performance

📊 Results
Model                     Accuracy
MLP                       ~97–98%
Random Forest             ~99–100%

🖥️ Dashboard

The Streamlit dashboard provides:

* Dataset statistics
* Malware vs Benign distribution
* Model comparison
* Accuracy metrics
* ROC curve visualization
* Prediction confidence scores

📈 Future Improvements

* Real-time malware detection
* Integration with SIEM/EDR systems
* CNN and LSTM-based models
* Federated learning support
* Adversarial robustness testing
* Cloud deployment

⸻

📚 References

[1] Carrier, T., Victor, P., Tekeoglu, A., & Lashkari, A. H. Detecting Obfuscated Malware using Memory Feature Engineering, ICISSP 2022.

[2] CIC-MalMem-2022 Dataset, Canadian Institute for Cybersecurity.

[3] Breiman, L. Random Forests, Machine Learning, 2001.

[4] Kingma, D. P., & Ba, J. Adam: A Method for Stochastic Optimization, ICLR 2015.

[5] Paszke, A. et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library, NeurIPS 2019.



