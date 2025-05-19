// pcm-processor.js - place in public/ directory

class PCMProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    // options.processorOptions.targetSampleRate contains the desired sample rate (e.g., 16000)
    // For this basic version, we assume the inputNode to this processor already provides the correct sample rate.
    // A more advanced version would handle resampling here if currentSampleRate !== targetSampleRate.
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

    // Convert Float32Array to Int16Array (PCM16)
    // Input samples are in the range [-1.0, 1.0]
    // Output samples should be in the range [-32768, 32767]
    const pcm16Data = new Int16Array(inputChannelData.length);
    for (let i = 0; i < inputChannelData.length; i++) {
      let s = Math.max(-1, Math.min(1, inputChannelData[i]));
      pcm16Data[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }

    // Post the Int16Array buffer back to the main thread
    // Transferable objects (ArrayBuffer) are sent by reference, not copied.
    this.port.postMessage(pcm16Data.buffer, [pcm16Data.buffer]);

    return true; // Keep processor alive
  }
}

registerProcessor('pcm-processor', PCMProcessor); 