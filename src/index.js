const { classifyInputForClient } = require('./classifier');
const { fineTuneModel, saveModelToLocalStorage, loadModelFromLocalStorage } = require('./model');
const { preprocessInput } = require('./preprocess');

// Export everything that needs to be available publicly in your package
module.exports = {
    classifyInputForClient,
    fineTuneModel,
    saveModelToLocalStorage,
    loadModelFromLocalStorage,
    preprocessInput
};
