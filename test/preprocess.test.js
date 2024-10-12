const { preprocessInput } = require('../src/preprocess');

describe('Input Preprocessing', () => {
    test('Preprocesses input by cleaning and lowercasing', () => {
        const rawInput = '  Machine Learning ENGINEER! ';
        const preprocessed = preprocessInput(rawInput);

        // Check that the input is correctly lowercased and cleaned
        expect(preprocessed).toBe('machine learning engineer');
    });

    test('Removes special characters and trims spaces', () => {
        const rawInput = 'Java$$ Developer@@  ';
        const preprocessed = preprocessInput(rawInput);

        // Check that special characters are removed and spaces are trimmed
        expect(preprocessed).toBe('java developer');
    });
});
