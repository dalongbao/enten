class FishBrain {
    constructor() {
        this.session = null;
        this.isReady = false;
        this.modelPath = null;
    }

    async load(modelPath) {
        this.modelPath = modelPath;
        
        try {
            this.session = await ort.InferenceSession.create(modelPath, {
                executionProviders: ['webgl', 'wasm'],
                graphOptimizationLevel: 'all',
            });
            
            this.isReady = true;
            console.log(`FishBrain: Model loaded from ${modelPath}`);
            console.log(`  Input: ${this.session.inputNames}`);
            console.log(`  Output: ${this.session.outputNames}`);
            
        } catch (error) {
            console.error('FishBrain: Failed to load model:', error);
            throw error;
        }
    }

    async infer(observation) {
        if (!this.isReady) {
            throw new Error('FishBrain: Model not loaded. Call load() first.');
        }

        if (observation.length !== 51) {
            throw new Error(`FishBrain: Expected 51 observations, got ${observation.length}`);
        }

        const inputTensor = new ort.Tensor('float32', observation, [1, 51]);

        const feeds = { observation: inputTensor };
        const results = await this.session.run(feeds);

        const actionData = results.action.data;
        
        return {
            thrust: actionData[0],
            turn: actionData[1],
        };
    }

    async inferBatch(observations, batchSize) {
        if (!this.isReady) {
            throw new Error('FishBrain: Model not loaded. Call load() first.');
        }

        const expectedLength = batchSize * 51;
        if (observations.length !== expectedLength) {
            throw new Error(`FishBrain: Expected ${expectedLength} values, got ${observations.length}`);
        }

        const inputTensor = new ort.Tensor('float32', observations, [batchSize, 51]);

        const feeds = { observation: inputTensor };
        const results = await this.session.run(feeds);

        const actionData = results.action.data;
        const actions = [];
        
        for (let i = 0; i < batchSize; i++) {
            actions.push({
                thrust: actionData[i * 2],
                turn: actionData[i * 2 + 1],
            });
        }
        
        return actions;
    }

    ready() {
        return this.isReady;
    }

    async dispose() {
        if (this.session) {
            await this.session.release();
            this.session = null;
            this.isReady = false;
            console.log('FishBrain: Model disposed');
        }
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { FishBrain };
} else if (typeof window !== 'undefined') {
    window.FishBrain = FishBrain;
}
