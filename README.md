# 🧠 Linear Classifier Practice
Hi...  
In this mini-project I worked with a dataset that has 100 samples, 4 features, and 2 classes.  
The goal is to apply feature selection and train a linear classifier like Perceptron or least squares.  
<br>

We do it step-by-step in 3 stages... Check them below! 👇
<br>

## :blush:Step One: (Data Onboarding & Preparation):blush:</b>
First, we load the data and prepare it for classification.
<br>
📦 The dataset contains:  
- 100 samples  
- 4 features  
- 2 classes
  
<br>
We may also extract or generate new features if needed *(optional)*  
And finally, split the dataset for training and evaluation.
<br>
**Data files:**  
- data.csv  
- data.mat  
<br>

## :blush:Step Two: (Feature Selection):blush:</b>
Next, we select two features from the dataset for classification.
<br>

🔢 Total combinations of 2 features from 4 is: C(4, 2) = 6
<br>

We go through these steps:
1. **Feature combinations** – List all 2-feature combos  
2. **Feature evaluation** – Train a model on each pair and measure accuracy  
3. **Feature selection** – Pick the pair that performs best  
4. **Data normalisation (optional)** – Apply scaling if needed
<br>

We used Python to evaluate combinations and choose the best feature pair 💪
<br>

## :blush:Step Three: (Linear Classification):blush:</b>
We implement a linear classifier using:
🧮 **Perceptron loss** or **Sum of Squared Error (SSE)**  
<br>

Tasks in this part:
1. **Data recoding** – Add bias term (constant 1) to each input  
2. **Parameter estimation** – Learn weights  
3. **Linear classification** – Predict class labels  
4. **Performance evaluation** – Check test accuracy  
5. **Accuracy:** If you train and test on the same dataset, you'll get 100% accuracy (⚠️ Overfitting alert!)
<br>

🧠 Note: always split the data or use cross-validation for reliable results!
<br>
