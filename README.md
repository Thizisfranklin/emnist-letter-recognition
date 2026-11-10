# EMNIST HANDWRITERN CHARACTER RECOGNITION
> This is a computer vision work on recognizing handwritten characters using deep learning  
> We are using a subset of the extended popular MNIST dataset, named as EMNIST  
> This particular data variant has 10,000 samples and had its own random spilt for train and test 

[Cleaned and Preprocessed Data via Google Drive](https://drive.google.com/drive/folders/1Lb3EGMDdHWG85wYMiCmRS2rJZSia8MKU?usp=sharing)

### OVERVIEW: 
MNIST is a widely used benchmark for handwritten-digit recognition. EMNIST extends this benchmark design to handwritten letters and digits using data from NIST Special Database 19.

Our project uses a 10,000-image subset of EMNIST Letters, covering 26 classes with uppercase and lowercase forms grouped together. During preprocessing, we found that the supplied test subset did not contain all 26 letters. We therefore reserved it and created stratified training, validation, and test partitions from the source-training images.

We will compare a fully connected neural-network baseline with two CNN architectures. We also plan to evaluate separately collected handwriting to investigate how well the models generalize beyond our benchmark subset.


## Contributors

- Nathan Butler
- Michael Sheehey
- Franklin Osualaaham
- Joseph Uzoigwe
- Sarthakkumar Patel
