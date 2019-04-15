
#### Intro
- This is being under construction  

- This is my study using tutorial which is originated from  
https://github.com/Keess324/Multi-label-Image-Classification-  

- To have prerequisites on dataset, files, reviewing order, etc, read following file in the project  
`Instructions of Reading Files.docx`

- I translate original Keras code into PyTorch code with adding my functions.  

- Project and its dataset are from  
https://www.kaggle.com/c/planet-understanding-the-amazon-from-space
You need to download full 30 GB dataset from that page
and also need to uncompress files and put them into `Data` directory

#### Libraries
- TensorFlow: v1.10.0  
- Keras: v2.2.4  
- Pytorch: v1.0.0
- CUDA: V10.0.130  
- cuDNN: v7.4.2  
- Tqdm  
- etc

#### Used techniques
- Multi labels (for example, image1 can have 3 labels)  
- XGBoost  
- CNN  
- SIFT  

#### Metrics
1. Exact Match Ratio (MR)  
2. Hamming Loss (HL)  
3. Godbole et Measure (Considering the partical correct)  
