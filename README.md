# one-more-validation - npm Package

The **one-more-validation** is an npm package that helps validate input strings, such as job titles, task names, or positions, determining if they are relevant and suitable (e.g., 'Java Developer', 'ML Engineer') or irrelevant (e.g., 'xyz', 'I love you'). This package provides a prediction along with a confidence score to indicate how well the input fits the context of a job or position.

## Features

- Validates user-provided input to determine if it is relevant as a job title, task name, or similar.
- Uses machine learning to provide a confidence score for each input.
- Works on both the backend (Node.js) and frontend (React.js).
- Ability to fine-tune with client-specific data for customized predictions.

## Installation

Install the package via npm:

```bash
npm install one-more-validation
```

## Usage

Here is an example of how to use **one-more-validation**:

### Backend Example (Node.js)

```js
const { classifyInputForClient } = require('one-more-validation');

const clientId = 'client123';
const clientData = [
  { title: 'Software Engineer', label: 'Good Fit' },
  { title: 'Web Developer', label: 'Good Fit' },
  { title: 'xyz', label: 'Irrelevant' }
];

async function runClassification() {
  const input = 'Java Developer';
  const result = await classifyInputForClient(clientId, input, clientData);
  console.log(`Input: "${input}", Result: ${result.result}, Confidence: ${result.confidence}`);
}

runClassification();
```

### Frontend Example (React.js)

```jsx
import React, { useState } from 'react';
import { classifyInputForClient } from 'one-more-validation';

function JobTitleClassifier() {
  const [input, setInput] = useState('');
  const [result, setResult] = useState(null);

  const clientId = 'client123';
  const clientData = [
    { title: 'Software Engineer', label: 'Good Fit' },
    { title: 'Web Developer', label: 'Good Fit' },
    { title: 'xyz', label: 'Irrelevant' }
  ];

  const handleClassify = async () => {
    const classificationResult = await classifyInputForClient(clientId, input, clientData);
    setResult(classificationResult);
  };

  return (
    <div>
      <h1>Job Title Classifier</h1>
      <input
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder="Enter job title"
      />
      <button onClick={handleClassify}>Classify</button>
      {result && (
        <div>
          <p>Result: {result.result}</p>
          <p>Confidence: {result.confidence}</p>
        </div>
      )}
    </div>
  );
}

export default JobTitleClassifier;
```

## Methods

### `classifyInputForClient(clientId, input, clientData)`

- **clientId** (string): Identifier for the client.
- **input** (string): The text input to classify (e.g., job title).
- **clientData** (array): Client-specific training data for fine-tuning.
- **Returns**: An object containing `result` (classification result: "Good Fit" or "Irrelevant") and `confidence` (confidence score).

### `fineTuneModel(clientData, modelSavePath)`

- Fine-tunes the classifier using the provided client-specific data.

### `saveModelToFileSystem(model, savePath)`

- Saves the fine-tuned model to the file system.

### `loadModelFromFileSystem(savePath)`

- Loads the fine-tuned model from the file system if it exists.

## How It Works

- The package uses the **Universal Sentence Encoder** to embed input strings into numerical vectors.
- A fine-tuned **TensorFlow.js** model is used to classify whether the input string is relevant or not.
- The model is trained on client-specific data to improve accuracy, allowing for customization based on client needs.

## License

MIT License

## Contributions

Contributions are welcome! Feel free to fork the repository and submit a pull request with your improvements or suggestions.

## Issues

If you encounter any issues, please open an issue in the GitHub repository.

## Author

**Amin** - Full Stack Developer and Machine Learning Specialist [LinkedIn](https://www.linkedin.com/in/followaamin/)