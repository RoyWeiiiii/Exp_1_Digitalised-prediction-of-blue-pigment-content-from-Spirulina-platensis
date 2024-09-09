# Graphical Abstract
![GA_Experiment_2](https://github.com/user-attachments/assets/b1b5a423-90f9-4705-8267-8a18a771108b)

# Digitalised prediction of blue pigment content from Spirulina platensis: Next-generation microalgae bio-molecule detection

The motive of this study is to predict the concentration of C-phycocyanin (CPC) from _Spirulina platensis_ by adapting several colour models along with machine learning (ML) and deep learning (DL) techniques. Initially, three different culture mediums such as Zarrouk, BG-11, and AF6 were compared, and the BG-11 medium was chosen due to its overall best biomass growth, least amount of chemical usage, and CPC production. The performance of the convolutional neural network (CNN) without the input parameters of ‘Abs’ and ‘Day’ results in a higher R2 of 0.7269 as compared to both support vector machine (SVM) and artificial neural network (ANN) with R2 of 0.2725 and 0.2552, respectively. The absence of regularisation techniques has caused the scenario of model overfitting, showing results of R2Train = 0.9891 and R2Val = 0.5170 (without image augmentation) and R2Train = 0.9710 and R2Val = 0.5521 (including 20 % dropout but without image augmentation). Meanwhile, both SVM and ANN models were observed to show significantly high accuracy when including extra parameters of ‘Abs’ and ‘Day’ as compared to the CNN model with R2 of 0.9903 and 0.9827, respectively. We aim to establish a high precision and real-time assessment of microalgae biomolecule intelligent system that requires low cost, less time consumption, and is widely applicable, addressing the challenges associated with conventional microalgae quantification and identification.

**Keywords:** _Spirulina platensis_; C-phycocyanin; Colour feature; Machine learning (ML); Deep learning (DL)

# Folder and files description

**AI_models** => Contains the model configuration of ANN (specifically MLP) regressor, CNN (with data augmentation), CNN (without data augmentation), and SVM regressor

**Data_Features_ColourIndex_ANN_SVM** => Contains the combined data of all mediums (BG-11, AF6, & Zarrouk) and colourIndex features (RGB, HSL, & CMYK) for ANN (MLP) and SVM model

**Data_Features_ColourIndex_Day_Abs_ANN_SVM** => Contains the combined data of all mediums (BG-11, AF6, & Zarrouk), colourIndex features (RGB, HSL, & CMYK), Day (period), and Abs (Absorbance) for ANN (MLP) and SVM model

**RGB-HSL-CMYK_Extraction** => Contains the code configuration for the extraction of RGB, HSL, and CMYK colour features from _Spirulina platensis_ images

**Experiment_2_Tabulated_Data** => Contains all the results tabulated in excel format

# Referencing and citation
If you find the prediction and analysis of C-phycocyanin (CPC) concentration  useful in your research, please consider citing: Based on the DOI: https://doi.org/10.1016/j.algal.2024.103642
