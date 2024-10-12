// Preprocess input by cleaning and lowercasing
function preprocessInput(input) {
    return input.toLowerCase().replace(/[^a-z0-9 ]/g, '').trim();
}

module.exports = {
    preprocessInput
};
