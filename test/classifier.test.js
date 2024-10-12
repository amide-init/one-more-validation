const { LocalStorage } = require('node-localstorage');
const { classifyInputForClient } = require('../src/classifier');
const { fineTuneModel } = require('../src/model');

// Mock localStorage using node-localstorage
global.localStorage = new LocalStorage('./scratch');

// Example test data for classification
const clientData = [
    { title: 'Java Developer', label: 'Good Fit' },
    { title: 'Python Developer', label: 'Good Fit' },
    { title: 'I love you', label: 'Irrelevant' },
];

describe('Classifier Functionality', () => {
    test('Classify input as Good Fit or Irrelevant', async () => {
        const clientId = 'client123';
        const inputJobTitle = 'Machine Learning Engineer';

        const classification = await classifyInputForClient(clientId, inputJobTitle, clientData);

        // Verify that the classification result is as expected
        expect(classification).toHaveProperty('result');
        expect(classification).toHaveProperty('confidence');

        // We expect the result to be either 'Good Fit' or 'Irrelevant'
        expect(['Good Fit', 'Irrelevant']).toContain(classification.result);
        expect(classification.confidence).toBeGreaterThanOrEqual(0);
        expect(classification.confidence).toBeLessThanOrEqual(1);
    });

    test('Fine-tune model and classify input', async () => {
        const clientId = 'client456';
        const inputJobTitle = 'Frontend Developer';

        // Fine-tune a new model
        const model = await fineTuneModel(clientData, `${clientId}-model`);

        // Classify a new job title after fine-tuning
        const classification = await classifyInputForClient(clientId, inputJobTitle, clientData);

        // Verify that the classification result is as expected
        expect(classification).toHaveProperty('result');
        expect(classification.confidence).toBeGreaterThanOrEqual(0);
        expect(classification.confidence).toBeLessThanOrEqual(1);
    });
});
