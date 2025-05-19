'use client';

import { useState, useRef, useEffect } from 'react';

const PYTHON_WEBSOCKET_URL = 'ws://localhost:8765';
const TARGET_SAMPLE_RATE = 16000;

export default function HomePage() {
  const [isConnected, setIsConnected] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [status, setStatus] = useState('Idle');
  const [error, setError] = useState<string | null>(null);
  const [audioLevel, setAudioLevel] = useState(0);

  const socketRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const microphoneSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const pcmProcessorNodeRef = useRef<AudioWorkletNode | null>(null);
  const audioStreamRef = useRef<MediaStream | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const audioLevelTimerRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      // Cleanup resources on component unmount
      disconnectWebSocket();
      stopRecording(); // This will also handle AudioContext cleanup if active
      if (audioLevelTimerRef.current) {
        window.clearInterval(audioLevelTimerRef.current);
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const connectWebSocket = () => {
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      setStatus('Already connected to WebSocket.');
      return;
    }
    setStatus('Connecting to WebSocket...');
    setError(null);
    socketRef.current = new WebSocket(PYTHON_WEBSOCKET_URL);

    socketRef.current.onopen = () => {
      setIsConnected(true);
      setStatus('Connected to WebSocket. Ready to record.');
      console.log('WebSocket connected');
    };
    socketRef.current.onmessage = (event) => {
      console.log('Message from server:', event.data);
      setStatus(`Message from server: ${event.data}`);
    };
    socketRef.current.onerror = (event) => {
      console.error('WebSocket error:', event);
      setError(`WebSocket error. Check console. Is Python server at ${PYTHON_WEBSOCKET_URL} running?`);
      setStatus('WebSocket error.');
      setIsConnected(false);
    };
    socketRef.current.onclose = (event) => {
      setIsConnected(false);
      setIsRecording(false); 
      setStatus(`WebSocket disconnected: ${event.reason || 'No reason specified'}`);
      console.log('WebSocket disconnected:', event.reason);
      socketRef.current = null;
    };
  };

  const disconnectWebSocket = () => {
    if (socketRef.current) {
      setStatus('Disconnecting WebSocket...');
      socketRef.current.close();
    }
  };

  const startRecording = async () => {
    if (!isConnected || !socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
      setError('Not connected to WebSocket. Please connect first.');
      return;
    }
    if (isRecording) {
      setStatus('Already recording.');
      return;
    }

    setStatus('Requesting microphone access...');
    setError(null);

    try {
      // 1. Get microphone access and try to set desired sample rate
      // Note: browsers might not always honor the exact sampleRate.
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: TARGET_SAMPLE_RATE, // Request 16kHz
          channelCount: 1, // Request mono
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      audioStreamRef.current = stream;
      setStatus('Microphone access granted.');

      // 2. Create AudioContext
      // Try to create with the target sample rate. If not supported, browser uses its default.
      let context;
      try {
        context = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });
      } catch (e) {
        console.warn(`Failed to create AudioContext with ${TARGET_SAMPLE_RATE}Hz, using browser default. Error: ${e}`);
        context = new AudioContext(); // Fallback to browser default sample rate
      }
      audioContextRef.current = context;
      
      // Log the actual sample rate of the AudioContext
      console.log(`AudioContext sample rate: ${audioContextRef.current.sampleRate}Hz`);
      if (audioContextRef.current.sampleRate !== TARGET_SAMPLE_RATE) {
        const warningMsg = `AudioContext running at ${audioContextRef.current.sampleRate}Hz, not ${TARGET_SAMPLE_RATE}Hz. PCM Processor is NOT resampling. Deepgram might not process audio correctly.`;
        console.warn(warningMsg);
        setStatus(warningMsg + " Starting recording anyway.");
        // Do not set error here, allow testing connectivity but warn user.
      }

      // 3. Add AudioWorklet module
      try {
        await audioContextRef.current.audioWorklet.addModule('/pcm-processor.js');
      } catch (e) {
        console.error('Error adding audio worklet module', e);
        setError('Could not load audio processor. Is pcm-processor.js in public/ folder? Check console.');
        setStatus('Failed to load audio processor.');
        stream.getTracks().forEach(track => track.stop()); // Release mic
        if (audioContextRef.current.state !== 'closed') await audioContextRef.current.close();
        return;
      }
      setStatus('Audio processor loaded. Starting recording...');

      // 4. Create MediaStreamSource and AudioWorkletNode (PCM Processor)
      microphoneSourceRef.current = audioContextRef.current.createMediaStreamSource(stream);
      
      // Create analyzer for audio level visualization
      analyserRef.current = audioContextRef.current.createAnalyser();
      analyserRef.current.fftSize = 256;
      microphoneSourceRef.current.connect(analyserRef.current);
      
      pcmProcessorNodeRef.current = new AudioWorkletNode(audioContextRef.current, 'pcm-processor');
      
      // 5. Connect the nodes: microphone -> analyser -> pcmProcessor
      microphoneSourceRef.current.connect(pcmProcessorNodeRef.current);
      // The PCM processor does not output to speakers, so no pcmProcessorNode.connect(audioContext.destination)

      // 6. Handle messages (PCM data) from the PCMProcessor
      pcmProcessorNodeRef.current.port.onmessage = (event) => {
        const pcm16DataBuffer = event.data as ArrayBuffer;
        if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
          socketRef.current.send(pcm16DataBuffer);
        }
      };
      
      pcmProcessorNodeRef.current.port.onmessageerror = (error) => {
        console.error('Error message from PCM Processor:', error);
        setError('Error from audio processor. Check console.');
      };

      setIsRecording(true);
      setStatus('Recording and streaming PCM audio...');
      console.log('Recording started with AudioWorklet (PCMProcessor)');
      const audioTrackSettings = stream.getAudioTracks()[0].getSettings();
      console.log('Actual microphone audio track settings:', audioTrackSettings);
      if (audioTrackSettings.sampleRate && audioTrackSettings.sampleRate !== TARGET_SAMPLE_RATE) {
         console.warn(`Microphone is capturing at ${audioTrackSettings.sampleRate}Hz.`);
      }
      
      // Start monitoring audio levels
      startAudioLevelMonitoring();

    } catch (err) {
      console.error('Error starting AudioWorklet recording:', err);
      setError(`Error starting recording: ${err instanceof Error ? err.message : String(err)}`);
      setStatus('Failed to start recording.');
      setIsRecording(false);
      // Clean up any partial setup
      if (audioStreamRef.current) audioStreamRef.current.getTracks().forEach(track => track.stop());
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') audioContextRef.current.close();
    }
  };

  const stopRecording = async () => {
    setStatus('Stopping recording...');
    if (microphoneSourceRef.current) {
      microphoneSourceRef.current.disconnect();
      microphoneSourceRef.current = null;
    }
    if (analyserRef.current) {
      analyserRef.current.disconnect();
      analyserRef.current = null;
    }
    if (pcmProcessorNodeRef.current) {
      pcmProcessorNodeRef.current.disconnect();
      pcmProcessorNodeRef.current = null;
    }
    if (audioStreamRef.current) {
      audioStreamRef.current.getTracks().forEach(track => track.stop());
      audioStreamRef.current = null;
    }
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      try {
        await audioContextRef.current.close();
        console.log('AudioContext closed.');
      } catch (e) {
        console.error('Error closing AudioContext:', e);
      }
      audioContextRef.current = null;
    }
    setIsRecording(false);
    setStatus('Recording stopped.');
    console.log('Recording stopped.');
    
    // Stop audio level monitoring
    stopAudioLevelMonitoring();
  };

  // Function to update audio levels
  const startAudioLevelMonitoring = () => {
    if (analyserRef.current) {
      const updateAudioLevel = () => {
        const analyser = analyserRef.current;
        if (!analyser) return;
        
        const dataArray = new Uint8Array(analyser.frequencyBinCount);
        analyser.getByteFrequencyData(dataArray);
        
        // Calculate average level
        let sum = 0;
        for (let i = 0; i < dataArray.length; i++) {
          sum += dataArray[i];
        }
        const avgLevel = sum / dataArray.length;
        setAudioLevel(avgLevel);
      };
      
      // Update the level every 100ms
      audioLevelTimerRef.current = window.setInterval(updateAudioLevel, 100);
    }
  };

  const stopAudioLevelMonitoring = () => {
    if (audioLevelTimerRef.current) {
      window.clearInterval(audioLevelTimerRef.current);
      audioLevelTimerRef.current = null;
    }
    setAudioLevel(0);
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <h1>Real-time Voice Agent Test (AudioWorklet for PCM)</h1>
      
      <div>
        <strong>Status:</strong> {status}
      </div>
      {error && <div style={{ color: 'red' }}><strong>Error:</strong> {error}</div>}

      <hr style={{ margin: '20px 0' }} />

      <h2>WebSocket Connection</h2>
      {!isConnected ? (
        <button onClick={connectWebSocket} disabled={socketRef.current?.readyState === WebSocket.CONNECTING}>
          Connect to Python Server ({PYTHON_WEBSOCKET_URL})
        </button>
      ) : (
        <button onClick={disconnectWebSocket}>
          Disconnect from Server
        </button>
      )}
      <p>Connected: {isConnected ? 'Yes' : 'No'}</p>

      <hr style={{ margin: '20px 0' }} />

      <h2>Microphone Recording (PCM via AudioWorklet)</h2>
      <button onClick={startRecording} disabled={!isConnected || isRecording}>
        Start PCM Recording
      </button>
      <button onClick={stopRecording} disabled={!isRecording || !audioContextRef.current}>
        Stop PCM Recording
      </button>
      <p>Recording: {isRecording ? 'Yes' : 'No'}</p>
      
      {/* Audio Level Visualizer */}
      <div style={{ marginTop: '10px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div>Audio Level:</div>
          <div style={{ 
            width: '300px', 
            height: '20px', 
            border: '1px solid #ccc', 
            borderRadius: '3px',
            overflow: 'hidden'
          }}>
            <div style={{ 
              width: `${Math.min(100, (audioLevel / 255) * 100)}%`, 
              height: '100%', 
              backgroundColor: audioLevel > 200 ? '#ff0000' : audioLevel > 100 ? '#ffaa00' : '#00cc00',
              transition: 'width 0.1s ease-out'
            }} />
          </div>
          <div>{Math.round((audioLevel / 255) * 100)}%</div>
        </div>
      </div>
      
      {/* Audio Settings Info */}
      <div style={{ 
        marginTop: '20px', 
        padding: '10px', 
        backgroundColor: '#f0f0f0', 
        borderRadius: '5px',
        fontSize: '14px'
      }}>
        <h3 style={{ margin: '0 0 10px 0' }}>Audio Settings</h3>
        <p style={{ margin: '5px 0' }}>Target Sample Rate: {TARGET_SAMPLE_RATE}Hz</p>
        <p style={{ margin: '5px 0' }}>
          Actual Sample Rate: {audioContextRef.current ? `${audioContextRef.current.sampleRate}Hz` : 'Not initialized'}
          {audioContextRef.current && audioContextRef.current.sampleRate !== TARGET_SAMPLE_RATE && 
            <span style={{ color: 'red', marginLeft: '5px' }}>⚠️ Mismatch!</span>
          }
        </p>
        <p style={{ margin: '5px 0' }}>WebSocket URL: {PYTHON_WEBSOCKET_URL}</p>
      </div>

      <div style={{ marginTop: '30px', fontSize: '0.9em', color: '#555' }}>
        <p>
          <strong>Note:</strong> This version uses an AudioWorklet (<code>pcm-processor.js</code>) to attempt to stream raw 16-bit PCM audio at {TARGET_SAMPLE_RATE}Hz.
        </p>
        <p>
          The browser will attempt to use a {TARGET_SAMPLE_RATE}Hz sample rate for the microphone and AudioContext. 
          If your microphone or browser does not support this rate directly, the actual sample rate might differ. 
          The current <code>pcm-processor.js</code> <strong>now includes resampling</strong> to ensure correct audio format. 
          Check the browser console for logs on actual sample rates.
        </p>
        <p>
          Make sure <code>pcm-processor.js</code> is in your <code>public</code> Next.js folder.
        </p>
      </div>
    </div>
  );
} 