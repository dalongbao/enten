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

        if (observation.length !== 60) {
            throw new Error(`FishBrain: Expected 60 observations, got ${observation.length}`);
        }

        const inputTensor = new ort.Tensor('float32', observation, [1, 60]);

        const feeds = { observation: inputTensor };
        const results = await this.session.run(feeds);

        const actionData = results.action.data;

        // 6 fin actions: body_freq, body_amp, left_pec_freq, left_pec_amp, right_pec_freq, right_pec_amp
        return {
            body_freq: actionData[0],
            body_amp: actionData[1],
            left_pec_freq: actionData[2],
            left_pec_amp: actionData[3],
            right_pec_freq: actionData[4],
            right_pec_amp: actionData[5],
        };
    }

    async inferBatch(observations, batchSize) {
        if (!this.isReady) {
            throw new Error('FishBrain: Model not loaded. Call load() first.');
        }

        const expectedLength = batchSize * 60;
        if (observations.length !== expectedLength) {
            throw new Error(`FishBrain: Expected ${expectedLength} values, got ${observations.length}`);
        }

        const inputTensor = new ort.Tensor('float32', observations, [batchSize, 60]);

        const feeds = { observation: inputTensor };
        const results = await this.session.run(feeds);

        const actionData = results.action.data;
        const actions = [];

        for (let i = 0; i < batchSize; i++) {
            const offset = i * 6;
            actions.push({
                body_freq: actionData[offset],
                body_amp: actionData[offset + 1],
                left_pec_freq: actionData[offset + 2],
                left_pec_amp: actionData[offset + 3],
                right_pec_freq: actionData[offset + 4],
                right_pec_amp: actionData[offset + 5],
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
