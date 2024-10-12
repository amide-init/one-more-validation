const use = require('@tensorflow-models/universal-sentence-encoder');
const tf = require('@tensorflow/tfjs');

// Save the fine-tuned model to localStorage
async function saveModelToLocalStorage(model, modelName) {
    try {
        await model.save(`localstorage://${modelName}`);
        console.log(`Model saved to localStorage with the name ${modelName}`);
    } catch (error) {
        console.error('Error saving model to localStorage:', error);
    }
}

// Load the fine-tuned model from localStorage
async function loadModelFromLocalStorage(modelName) {
    try {
        const model = await tf.loadLayersModel(`localstorage://${modelName}`);
        console.log(`Model loaded from localStorage with the name ${modelName}`);
        return model;
    } catch (error) {
        console.error('Model not found in localStorage, fine-tuning required.', error);
        return null;
    }
}

// Fine-tune the model with client-specific data
async function fineTuneModel(clientData, modelName) {
    const modelUse = await use.load();
    const inputs = clientData.map(data => data.title.toLowerCase());
    const inputEmbeddings = await modelUse.embed(inputs);
    const labels = clientData.map(data => data.label === 'Good Fit' ? 1 : 0);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    const classifier = tf.sequential();
    classifier.add(tf.layers.dense({ units: 256, activation: 'relu', inputShape: [512] }));
    classifier.add(tf.layers.dropout({ rate: 0.3 }));
    classifier.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    classifier.compile({
        optimizer: tf.train.adam(0.00005),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy'],
    });

    await classifier.fit(inputEmbeddings, labelTensor, {
        epochs: 30,
        validationSplit: 0.2,
    });

    await saveModelToLocalStorage(classifier, modelName);

    inputEmbeddings.dispose();
    labelTensor.dispose();

    return classifier;
}

module.exports = {
    saveModelToLocalStorage,
    loadModelFromLocalStorage,
    fineTuneModel
};
