// pcm-processor.js - place in public/ directory

class PCMProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    // Store the actual context sample rate
    this.actualSampleRate = sampleRate;
    // Target sample rate for Deepgram (16000 Hz)
    this.targetSampleRate = 16000;
    // Resampling is needed if the rates don't match
    this.needsResampling = this.actualSampleRate !== this.targetSampleRate;
    
    // Simple resampling state
    this.resampleRatio = this.needsResampling ? this.targetSampleRate / this.actualSampleRate : 1;
    this.resampleBuffer = [];
    this.frameCounter = 0;
    
    console.log(`PCM Processor initialized. Actual sample rate: ${this.actualSampleRate}Hz, Target: 16000Hz, Resampling: ${this.needsResampling}, Ratio: ${this.resampleRatio}`);
    
    // Add additional debug output
    if (this.actualSampleRate !== this.targetSampleRate) {
      console.warn(`⚠️ Sample rate mismatch! Browser: ${this.actualSampleRate}Hz, Deepgram expects: ${this.targetSampleRate}Hz.`);
      console.log(`Using resampling ratio: ${this.resampleRatio}`);
    }
    
    this.port.onmessage = (event) => {
      // Handle messages from the main thread if needed in the future
      // console.log("[PCMProcessor] Message from main thread:", event.data);
    };
  }

  process(inputs, outputs, parameters) {
    // inputs[0] is an array of channels. inputs[0][0] is the Float32Array for the first channel.
    const inputChannelData = inputs[0][0];

    if (!inputChannelData) {
      return true; // Keep processor alive
    }
    
    this.frameCounter++;
    
    // Log audio levels occasionally to help debug
    if (this.frameCounter % 100 === 0) {
      let maxLevel = 0;
      for (let i = 0; i < inputChannelData.length; i++) {
        maxLevel = Math.max(maxLevel, Math.abs(inputChannelData[i]));
      }
      console.log(`Audio level: ${Math.round(maxLevel * 100)}% of max`);
    }

    // Step 1: Apply enhanced resampling if needed
    let processedData;
    if (this.needsResampling) {
      processedData = this.resample(inputChannelData);
    } else {
      processedData = inputChannelData;
    }
    
    // Step 2: Convert Float32Array to Int16Array (PCM16)
    // Input samples are in the range [-1.0, 1.0]
    // Output samples should be in the range [-32768, 32767]
    const pcm16Data = new Int16Array(processedData.length);
    for (let i = 0; i < processedData.length; i++) {
      // Ensure values are within [-1, 1] range
      let s = Math.max(-1, Math.min(1, processedData[i]));
      // Apply some light compression to boost quiet signals
      s = Math.sign(s) * Math.pow(Math.abs(s), 0.8);
      // Convert to 16-bit PCM range
      pcm16Data[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }

    // Post the Int16Array buffer back to the main thread
    // Transferable objects (ArrayBuffer) are sent by reference, not copied.
    this.port.postMessage(pcm16Data.buffer, [pcm16Data.buffer]);

    return true; // Keep processor alive
  }
  
  /**
   * Enhanced resampling implementation with better interpolation
   */
  resample(inputBuffer) {
    const inputLength = inputBuffer.length;
    const outputLength = Math.round(inputLength * this.resampleRatio);
    const output = new Float32Array(outputLength);
    
    // Enhanced resampling with cubic interpolation for better quality
    for (let i = 0; i < outputLength; i++) {
      const exactInputIndex = i / this.resampleRatio;
      const inputIndex = Math.floor(exactInputIndex);
      const fraction = exactInputIndex - inputIndex;
      
      if (inputIndex >= inputLength - 1) {
        // Handle edge case at the end of the buffer
        output[i] = inputBuffer[inputLength - 1];
      } else {
        // Use cubic interpolation when possible (need 4 points)
        if (inputIndex > 0 && inputIndex < inputLength - 2) {
          const y0 = inputBuffer[inputIndex - 1];
          const y1 = inputBuffer[inputIndex];
          const y2 = inputBuffer[inputIndex + 1];
          const y3 = inputBuffer[inputIndex + 2];
          
          // Cubic interpolation formula
          const a0 = y3 - y2 - y0 + y1;
          const a1 = y0 - y1 - a0;
          const a2 = y2 - y0;
          const a3 = y1;
          
          output[i] = a0 * Math.pow(fraction, 3) + a1 * Math.pow(fraction, 2) + a2 * fraction + a3;
        } else {
          // Fall back to linear interpolation when cubic isn't possible
          output[i] = inputBuffer[inputIndex] * (1 - fraction) + inputBuffer[inputIndex + 1] * fraction;
        }
      }
    }
    
    return output;
  }
}

registerProcessor('pcm-processor', PCMProcessor); 