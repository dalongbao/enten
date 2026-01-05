/**
 * Fish Brain - ONNX Runtime Web inference wrapper
 * 
 * Loads a trained policy network and provides inference for controlling the fish.
 * Uses ONNX Runtime Web for efficient browser-based neural network inference.
 * 
 * Model expects:
 *   Input: "observation" - Float32Array of 51 features
 *     - Raycasts (32): 16 rays * 2 (distance, intensity)
 *     - Lateral line (16): 8 sensors * 2 (pressure_x, pressure_y)
 *     - Proprioception (3): forward_vel, lateral_vel, angular_vel
 * 
 *   Output: "action" - Float32Array of 2 values
 *     - thrust: [0, 1] forward acceleration
 *     - turn: [-1, 1] rotation
 */

class FishBrain {
    constructor() {
        this.session = null;
        this.isReady = false;
        this.modelPath = null;
    }

    /**
     * Load the ONNX model from a URL or path.
     * @param {string} modelPath - Path to the .onnx model file
     * @returns {Promise<void>}
     */
    async load(modelPath) {
        this.modelPath = modelPath;
        
        try {
            // Create inference session with WebGL backend for GPU acceleration
            // Falls back to WASM if WebGL unavailable
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

    /**
     * Run inference to get action from observation.
     * @param {Float32Array} observation - 51-element observation vector
     * @returns {Promise<{thrust: number, turn: number}>} Action to take
     */
    async infer(observation) {
        if (!this.isReady) {
            throw new Error('FishBrain: Model not loaded. Call load() first.');
        }

        if (observation.length !== 51) {
            throw new Error(`FishBrain: Expected 51 observations, got ${observation.length}`);
        }

        // Create input tensor (batch size 1)
        const inputTensor = new ort.Tensor('float32', observation, [1, 51]);
        
        // Run inference
        const feeds = { observation: inputTensor };
        const results = await this.session.run(feeds);
        
        // Extract action output
        const actionData = results.action.data;
        
        return {
            thrust: actionData[0],  // [0, 1]
            turn: actionData[1],    // [-1, 1]
        };
    }

    /**
     * Run batched inference for multiple observations.
     * @param {Float32Array} observations - Flat array of N*51 observations
     * @param {number} batchSize - Number of observations in batch
     * @returns {Promise<Array<{thrust: number, turn: number}>>} Array of actions
     */
    async inferBatch(observations, batchSize) {
        if (!this.isReady) {
            throw new Error('FishBrain: Model not loaded. Call load() first.');
        }

        const expectedLength = batchSize * 51;
        if (observations.length !== expectedLength) {
            throw new Error(`FishBrain: Expected ${expectedLength} values, got ${observations.length}`);
        }

        // Create input tensor
        const inputTensor = new ort.Tensor('float32', observations, [batchSize, 51]);
        
        // Run inference
        const feeds = { observation: inputTensor };
        const results = await this.session.run(feeds);
        
        // Extract actions
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

    /**
     * Check if model is loaded and ready.
     * @returns {boolean}
     */
    ready() {
        return this.isReady;
    }

    /**
     * Dispose of the model and free resources.
     */
    async dispose() {
        if (this.session) {
            await this.session.release();
            this.session = null;
            this.isReady = false;
            console.log('FishBrain: Model disposed');
        }
    }
}

// Export for use as ES module or global
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { FishBrain };
} else if (typeof window !== 'undefined') {
    window.FishBrain = FishBrain;
}
