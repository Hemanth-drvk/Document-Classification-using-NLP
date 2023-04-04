# Document Classification (NLP) using Decision Trees and Naive Bayes
### Overview 
This Python script performs document classification on a subset of the 20 newsgroups dataset. The script uses a Decision Tree classifier to classify the documents into five categories: alt.atheism, comp.graphics, rec.motorcycles, sci.space, and talk.politics.guns. The script utilizes the TfidfVectorizer from scikit-learn library to convert the raw text documents to a matrix of TF-IDF features. The TfidfVectorizer function takes care of removing stop words and other preprocessing tasks.

### Requirements
The code requires the following :
- Python 3.6 or higher
- NumPy
- scipy
- matplotlib
- pandas
- scikit-learn  

### Running the code
The code is available in a multiple files, Document_Classification_NB.py is running a Naive Bayes Classifier and Document_Classification_IDT.py is running a Decision Tree Classifier. You can run the code using the following command:

```bash
python Document_Classification_NB.py
python Document_Classification_IDT.py

```
### Code Structure 
The code consists of the following parts:

- Loading the data: The code loads the 20 newsgroups dataset and filters the documents belonging to the five categories mentioned above.
- Data preprocessing: The code uses the TfidfVectorizer function to convert the raw text documents to a matrix of TF-IDF features.
- Building the model: The code uses a Decision Tree classifier to build the classification model.
- Evaluating the model: The code evaluates the performance of the classification model on the train and independent test sets and displays the classification metrics.

### Output 
The output of the code includes the following:

- The train and test accuracy scores of the classification model.
- The classification report containing precision, recall, and F1-score for each class.
- The classification of a test example as an example of how the model works.
