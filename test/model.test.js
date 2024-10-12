const { LocalStorage } = require('node-localstorage');
const { saveModelToLocalStorage, loadModelFromLocalStorage } = require('../src/model');

// Mock localStorage using node-localstorage
global.localStorage = new LocalStorage('./scratch');

// Mock model data (You can replace this with real TensorFlow.js models)
const mockModel = {
    toJSON: () => ({ name: 'mock-model', layers: [] }),
    save: jest.fn(),
    getWeights: jest.fn(() => [])
};

describe('Model LocalStorage Functionality', () => {
    test('Save model to localStorage', async () => {
        const modelName = 'testModel';

        // Save the model to localStorage
        await saveModelToLocalStorage(mockModel, modelName);

        // Check that the model is saved in localStorage
        const savedModel = localStorage.getItem(`tensorflowjs_models/${modelName}/info`);
        expect(savedModel).toBeDefined();
    });

    test('Load model from localStorage', async () => {
        const modelName = 'testModel';

        // Mock the save before trying to load
        await saveModelToLocalStorage(mockModel, modelName);

        // Load the model from localStorage
        const loadedModel = await loadModelFromLocalStorage(modelName);

        // Check that the model is loaded correctly
        expect(loadedModel).toBeDefined();
        expect(loadedModel).toHaveProperty('name', 'mock-model');
    });
});
