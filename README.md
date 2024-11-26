# Digit Recognition  

This project is a deep learning-based digit recognition model using the **MNIST dataset**. It classifies handwritten digits (0-9) and can predict digits from custom images. The model is built using **TensorFlow** and **Keras** and trained on the MNIST dataset.  

## Overview  

The **MNIST dataset** is a widely used benchmark for handwritten digit classification, containing:  
- **60,000 training images**  
- **10,000 test images**, each labeled from 0 to 9.  

This project demonstrates:  
1. **Loading and visualizing data** from the MNIST dataset.  
2. **Building, training, and evaluating a neural network model** for digit recognition.  
3. **Using the trained model to predict digits** from new, user-provided images.  

## Tools & Libraries  

- **Numpy**: For numerical operations and data manipulation.  
- **Matplotlib & Seaborn**: For data visualization and plotting the confusion matrix.  
- **OpenCV**: For image processing and custom image input handling.  
- **PIL (Python Imaging Library)**: To manage image file formats.  
- **TensorFlow & Keras**: For building and training the neural network model.  

## Project Structure  

1. **Load and Explore Data**:  
   - Loads the MNIST dataset and explores the structure of training and testing data.  

2. **Data Preprocessing**:  
   - Scales pixel values for better model performance.  

3. **Model Building**:  
   - Creates a neural network using Keras with three layers:  
     - **Flatten layer**: Converts 2D images into 1D arrays.  
     - **Two dense layers**: Uses ReLU activation for learning complex patterns.  
     - **Output layer**: Uses softmax activation for 10 classes (0-9).  

4. **Model Training and Evaluation**:  
   - Compiles the model using the **Adam optimizer**.  
   - Trains the model on the training dataset.  
   - Evaluates its accuracy on the test dataset.  

5. **Prediction on Custom Images**:  
   - Allows users to upload custom images of digits for prediction by the trained model.  

## Installation  

Follow these steps to set up and run the project locally:  

1. Clone this repository:  
   ```bash  
   git clone https://github.com/your-username/digit-recognition.git  
   cd digit-recognition
   ```
2. Create and activate a virtual environment (optional but recommended):
   ```bash  
   python -m venv env  
    env\Scripts\activate  
   ```
4. Install required dependencies:
   ```bash  
   pip install -r requirements.txt  
   ```
6. Run the project:
   ```bash  
   python main.py 
   ```
   
## Dataset
The project uses the MNIST dataset, which is automatically downloaded using TensorFlow/Keras utilities.

## Results
The model achieves high accuracy on the test dataset.
Sample evaluation metrics:
Accuracy: X%
Precision, Recall, F1-Score: See classification report


## Technologies Used
Python
TensorFlow/Keras (for neural network implementation)
Matplotlib/Seaborn (for visualizations)
OpenCV and PIL (for image handling)

# Future Improvements
Implement a more complex neural network (e.g., Convolutional Neural Network) for improved performance.
Add a user interface to make digit prediction more interactive.
Support additional datasets for broader applications.


# Contributing
Contributions are welcome! Feel free to fork this repository and submit a pull request.


# Acknowledgments
The MNIST dataset is provided by Yann LeCun and is widely used in deep learning projects.
Special thanks to the open-source community for providing tools and libraries that power this project.
