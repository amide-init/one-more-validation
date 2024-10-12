const { preprocessInput } = require('./preprocess');
const { loadModelFromLocalStorage, fineTuneModel } = require('./model');
const use = require('@tensorflow-models/universal-sentence-encoder');

// Classify input using the cached model, or fine-tune it if necessary
async function classifyInputForClient(clientId, input, clientData) {
    const modelName = `${clientId}-model`;

    let classifier = await loadModelFromLocalStorage(modelName);
    if (!classifier) {
        classifier = await fineTuneModel(clientData, modelName);
    }

    const modelUse = await use.load();
    const preprocessedInput = preprocessInput(input);
    const inputEmbedding = await modelUse.embed([preprocessedInput]);

    const prediction = classifier.predict(inputEmbedding);
    const confidence = prediction.dataSync()[0];

    inputEmbedding.dispose();
    prediction.dispose();

    return {
        confidence,
        result: confidence > 0.7 ? 'Good Fit' : 'Irrelevant'
    };
}

module.exports = {
    classifyInputForClient
};
