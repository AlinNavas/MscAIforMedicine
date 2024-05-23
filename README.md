# PROJECTS developed in Masters Program (2023-2024)

1. Music Genre Classification using Transfer Learning on Spectogram
    
This project attempts to classify different 30-second audio clips into 10 music
genre classes. We first use several preprocessing steps like s like Mel Spectrogram, Mel
Frequency Cepstral Coefficients, STFT (Short Time Fourier Transforms), Spectrogram to convert the audio files
into images and then attempt to perform the classification task initially using a
baseline CNN model and then using pre-trained models to gauge any performance
improvement with significant hyperparameter tuning using baysean optimisation.

2. Comparative Analysis of Clinical Validity of ECG Explainability Methods Across Diverse Neural Network Architectures ( Code will be uploaded after publication )

Aims: In the current literature, there are several deep learning architectures of comparable performance that employ different neural network architectures to predict cardiac conditions from ECG signals. Besides the obvious difference in computation cost for the different models, I wanted to utilize different explainability methods to identify the method that most closely reflects clinical logic and whether some of these models are more focused than others on relevant clinical features for the different pathological beats.

Methods and results: I have used a publicly available MIT-BIH dataset for the analysis. The data was pre-processed and then fed into algorithms of comparable f1 scores designed by different researchers using different architectures like MLP, CNN, and LSTM algorithms. Explainability methods like PFI, SHAP, LIME, and Grad CAM were used to identify the important segments for each pathological class. The results were assessed by evaluating them against the rule-based approach designed from clinical texts.




