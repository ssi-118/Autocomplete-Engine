
# Autocomplete Engine - FlowType

This project is a simple NLP-based autocomplete web application built using **PyTorch**, **LSTM**, and **Streamlit**. The system suggests word and sentence completions while the user types, similar to a search bar autocomplete feature.

## Features

- Real-time autocomplete suggestions
- Partial word completion
- Next-word prediction using an LSTM model
- Streamlit-based web interface
- Dropdown-style suggestions
- Clickable suggestions that update the input text

## Technologies Used

- Python
- PyTorch
- Streamlit
- LSTM Neural Network
- NLP text preprocessing
- Google Colab for training

## Project Structure

```text
project-folder/
│
├── src/
│   ├── app.py
│   └── autocomplete_lstm.pth
│
├── assets/
│   └── Autocomplete.ipynb
│
├── docs/
│   └── Autocomplete_IP_OP.docx
│
├── requirements.txt
└── README.md
```

## Dataset

The model is trained using a cleaned text dataset saved as:

```text
output.csv
```

The dataset contains English sentences in a column named:

```text
Text
```

Example:

```csv
Text
artificial intelligence is changing the world
machine learning is used in many applications
natural language processing helps computers understand text
```

## Model

The project uses a word-level LSTM model. The model learns from sequences of words and predicts the next possible word based on the previous words.

The trained model is saved as:

```text
autocomplete_lstm.pth
```

## Installation

Clone or download the project folder.

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Requirements

```txt
streamlit
torch
streamlit-searchbox
```

## Running the Application

From the project root folder, run:

```bash
streamlit run src/app.py
```

The app will open in the browser.

## Example Input and Output

```text
Input:
artificial intell

Output Suggestions:
1. artificial intelligence
2. artificial intellectual
3. artificial intelligent
```

## How It Works

1. The user starts typing in the input box.
2. If the user is typing a partial word, the system gives matching word suggestions from the vocabulary.
3. If the phrase is complete, the LSTM model predicts the next possible word.
4. Suggestions are shown in a dropdown.
5. When the user selects a suggestion, it is added to the input box.
6. The user can continue typing from the selected suggestion.

## Training Workflow

The model is trained in Google Colab using PyTorch.

Basic workflow:

```text
Dataset → Text Cleaning → Vocabulary Creation → Sequence Generation → LSTM Training → Save Model
```

After training, the model is downloaded as:

```text
autocomplete_lstm.pth
```

This file is placed inside the `src` folder with `app.py`.

## Live Demo
```
https://autocomplete-engine-hxxfaivcdb8w56sts2vehs.streamlit.app/
```