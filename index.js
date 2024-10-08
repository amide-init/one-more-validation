const use = require('@tensorflow-models/universal-sentence-encoder');
const tf = require('@tensorflow/tfjs');
const fs = require('fs').promises;
const path = require('path');

// Preprocess input by cleaning and lowercasing
function preprocessInput(input) {
    return input.toLowerCase().replace(/[^a-z0-9 ]/g, '').trim();
}

// Fine-tune the model in Node.js with client-specific data
async function fineTuneModel(clientData, modelSavePath) {
    const modelUse = await use.load(); // Load the Universal Sentence Encoder

    // Convert client data into embeddings
    const inputs = clientData.map(data => preprocessInput(data.title));
    const inputEmbeddings = await modelUse.embed(inputs); // Shape: [num_samples, 512]

    // Create labels (1 = relevant, 0 = irrelevant)
    const labels = clientData.map(data => data.label === 'Good Fit' ? 1 : 0);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]); // Shape: [num_samples, 1]

    // Balance the dataset if necessary
    balanceDataset(clientData);

    // Simple classifier model to fine-tune
    const classifier = tf.sequential();
    classifier.add(tf.layers.dense({ units: 256, activation: 'relu', inputShape: [512] }));
    classifier.add(tf.layers.dropout({ rate: 0.3 }));
    classifier.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    classifier.add(tf.layers.dropout({ rate: 0.3 }));
    classifier.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    classifier.add(tf.layers.dropout({ rate: 0.3 }));
    classifier.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
    classifier.compile({
        optimizer: tf.train.adam(0.00005),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy'],
    });

    // Initialize variables to track the best model
    let bestValLoss = Infinity;
    let bestEpoch = -1;
    let patienceCounter = 0;
    const patience = 5; // Number of epochs to wait before stopping

    // Train the classifier in Node.js
    try {
        await classifier.fit(inputEmbeddings, labelTensor, {
            epochs: 30,
            validationSplit: 0.2,
            callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    console.log(`Epoch ${epoch + 1}: Loss=${logs.loss.toFixed(4)}, Val Loss=${logs.val_loss.toFixed(4)}, Accuracy=${logs.acc.toFixed(4)}, Val Accuracy=${logs.val_acc.toFixed(4)}`);

                    if (logs.val_loss < bestValLoss) {
                        bestValLoss = logs.val_loss;
                        bestEpoch = epoch + 1;
                        patienceCounter = 0; // Reset patience counter

                        // Save the model as it has the best validation loss so far
                        await saveModelToFileSystem(classifier, modelSavePath);
                        console.log(`Best model updated at epoch ${bestEpoch} with Val Loss=${bestValLoss.toFixed(4)}`);
                    } else {
                        patienceCounter += 1;
                        console.log(`No improvement in Val Loss. Patience counter: ${patienceCounter}/${patience}`);

                        if (patienceCounter >= patience) {
                            console.log(`Early stopping triggered after ${patience} epochs with no improvement.`);
                            throw new Error('EarlyStopping: Training halted due to no improvement in validation loss.');
                        }
                    }
                },
            },
        });
    } catch (error) {
        if (error.message.startsWith('EarlyStopping')) {
            console.log('Early stopping has been triggered.');
        } else {
            console.error('An unexpected error occurred during training:', error);
        }
    }

    // Dispose tensors to free memory
    inputEmbeddings.dispose();
    labelTensor.dispose();

    console.log(`Training completed. Best Val Loss=${bestValLoss.toFixed(4)} at epoch ${bestEpoch}.`);
    return classifier;
}

// Balance the dataset to have equal number of 'Good Fit' and 'Irrelevant' labels
function balanceDataset(clientData) {
    const goodFitCount = clientData.filter(data => data.label === 'Good Fit').length;
    const irrelevantCount = clientData.filter(data => data.label === 'Irrelevant').length;

    if (goodFitCount === irrelevantCount) {
        return;
    }

    if (goodFitCount > irrelevantCount) {
        const difference = goodFitCount - irrelevantCount;
        const irrelevantSamples = clientData.filter(data => data.label === 'Irrelevant');
        for (let i = 0; i < difference; i++) {
            // Clone objects to avoid reference issues
            clientData.push({ ...irrelevantSamples[i % irrelevantSamples.length] });
        }
    } else {
        const difference = irrelevantCount - goodFitCount;
        const goodFitSamples = clientData.filter(data => data.label === 'Good Fit');
        for (let i = 0; i < difference; i++) {
            // Clone objects to avoid reference issues
            clientData.push({ ...goodFitSamples[i % goodFitSamples.length] });
        }
    }

    console.log(`Balanced dataset: Good Fit=${clientData.filter(data => data.label === 'Good Fit').length}, Irrelevant=${clientData.filter(data => data.label === 'Irrelevant').length}`);
}

// Save the fine-tuned model to the file system manually
async function saveModelToFileSystem(model, savePath) {
    // Ensure the directory exists
    const saveDir = path.resolve(savePath);
    await fs.mkdir(saveDir, { recursive: true });

    // Get model architecture
    const modelTopology = model.toJSON();

    // Save model architecture to model.json
    await fs.writeFile(path.join(saveDir, 'model.json'), JSON.stringify(modelTopology));

    // Get model weights
    const weights = model.getWeights();
    const weightsData = [];

    for (let i = 0; i < weights.length; i++) {
        const tensor = weights[i];
        const buffer = await tensor.data();
        weightsData.push(Array.from(buffer));
    }

    // Save weights to weights.json
    await fs.writeFile(path.join(saveDir, 'weights.json'), JSON.stringify(weightsData));

    console.log(`Model saved to ${saveDir}`);
}

// Load the fine-tuned model from the file system manually
async function loadModelFromFileSystem(savePath) {
    try {
        const modelJsonPath = path.join(savePath, 'model.json');
        const weightsJsonPath = path.join(savePath, 'weights.json');

        // Check if files exist
        await fs.access(modelJsonPath);
        await fs.access(weightsJsonPath);

        // Load model architecture
        const modelTopology = JSON.parse(await fs.readFile(modelJsonPath, 'utf8'));
        const model = await tf.models.modelFromJSON(modelTopology);

        // Load weights
        const weightsData = JSON.parse(await fs.readFile(weightsJsonPath, 'utf8'));
        const weights = weightsData.map(wData => tf.tensor(wData));

        // Set weights
        model.setWeights(weights);

        console.log(`Model loaded from ${savePath}`);
        return model;
    } catch (error) {
        console.error('Model not found in the file system, fine-tuning required.', error);
        return null;
    }
}

// Classify input using the cached model, or fine-tune it if necessary
async function classifyInputForClient(clientId, input, clientData) {
    const modelSavePath = path.join(__dirname, 'models', `${clientId}-model`);

    // Try to load the fine-tuned classifier model from the file system
    let classifier = await loadModelFromFileSystem(modelSavePath);

    // If no classifier is found, fine-tune a new one and save it
    if (!classifier) {
        classifier = await fineTuneModel(clientData, modelSavePath);
    }

    // Load the Universal Sentence Encoder
    const modelUse = await use.load();

    // Preprocess the input
    const preprocessedInput = preprocessInput(input);

    // Embed the preprocessed input
    const inputEmbedding = await modelUse.embed([preprocessedInput]); // Shape: [1, 512]

    // Predict using the classifier
    const prediction = classifier.predict(inputEmbedding);

    // Extract the confidence score
    const confidence = prediction.dataSync()[0];

    // Dispose tensors to free memory
    inputEmbedding.dispose();
    prediction.dispose();

    // Determine the classification based on confidence
    return {
        confidence: confidence, 
        result: confidence > 0.7 ? 'Good Fit' : "Irrelevant"
    };
}

module.exports = {
    classifyInputForClient,
    fineTuneModel,
    saveModelToFileSystem,
    loadModelFromFileSystem
};

    



