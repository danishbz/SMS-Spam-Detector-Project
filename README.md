# SMS Spam Detector Project

## Overview

This project focuses on developing a robust SMS spam detection system using natural language processing (NLP) techniques. Our goal is to mitigate the risks posed by spam SMS messages, which frequently lead to scams compromising personal and financial information. By creating a user-friendly tool, we aim to not only identify spam messages but also educate users, particularly the elderly and less tech-savvy individuals, on how to recognize and deal with spam in the future.

## Dataset

- **Total Messages**: 5,574 SMS messages categorized into spam or legitimate (ham).
- **Sources**:
    - **Grumbletext**: 425 spam messages from a UK forum.
    - **NUS SMS Corpus**: 3,375 legitimate messages from Singapore, primarily university students.
    - **Caroline Tag’s PhD Thesis**: 450 legitimate messages.
    - **SMS Spam Corpus v.0.1 Big**: 1,002 legitimate and 322 spam messages.

## Methodology

The dataset was split into 80% training and 20% testing data to ensure the model's effectiveness. The model leverages:

- Technologies:
    - Python Libraries: numpy, pandas, scikit-learn.
    - TfidfVectorizer for text feature extraction.
    - LogisticRegression for prediction modeling.

## Results

- Accuracy: Achieved over 90% accuracy, with 96.7% on training data and 96.5% on testing data, indicating an excellent detection capability.
- Evaluation: The model exhibits consistent performance without signs of overfitting or undertraining.

## Future Directions

- Extend the application to email spam detection.
- Develop mobile applications using:
    - iOS: Swift programming language.
    - Android: Kotlin programming language.

By deploying this model on mobile platforms, users can maintain security against spam across different devices.

## Usage

To run the SMS spam detector:

1. Ensure Python and required libraries (numpy, pandas, sklearn) are installed.
2. Import libraries and load mail_data.csv.
3. Train the model using the dataset.
4. Use the model to detect spam in new SMSes.

```
# Required installations
# Run: pip install numpy pandas scikit-learn

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('mail_data.csv')
# Follow up with data processing and model training...
```

## Conclusion

This project marks a significant step towards enhancing SMS security and user education. By identifying and understanding spam patterns, users gain valuable skills to navigate digital communications safely. Through future enhancements, this project's impact can encompass broader platforms, ensuring comprehensive protection against digital threats.

## License

This project is licensed under the MIT License.

You are free to use, modify, and distribute the code, provided that appropriate credit is given to the original authors and any changes are noted. This permit ensures that the project remains open-source and accessible while encouraging collaborative development and improvement.
