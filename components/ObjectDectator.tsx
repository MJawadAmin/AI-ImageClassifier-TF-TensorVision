// src/components/ObjectDetector.tsx
"use client";

import * as tf from '@tensorflow/tfjs'; // Add this
import { useEffect, useRef, useState } from 'react';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

// Force TensorFlow.js to use the CPU backend
tf.setBackend('cpu');

// Define types for better TypeScript support
interface Detection {
  class: string;
  score: number;
  bbox: [number, number, number, number]; // [x, y, width, height]
}

const ObjectDetector = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [detectedObjects, setDetectedObjects] = useState<Detection[]>([]);

  // Load TensorFlow model
  useEffect(() => {
    const loadModel = async () => {
      setIsModelLoading(true);
      try {
        const loadedModel = await cocoSsd.load();
        setModel(loadedModel);
        console.log('Model loaded successfully!');
      } catch (error) {
        console.error('Failed to load model:', error);
      } finally {
        setIsModelLoading(false);
      }
    };

    loadModel();
  }, []);

  // Setup webcam
  useEffect(() => {
    if (!model || isModelLoading) return;

    const setupCamera = async () => {
      if (!videoRef.current) return;
      
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'environment' },
          audio: false,
        });
        videoRef.current.srcObject = stream;
      } catch (error) {
        console.error('Error accessing webcam:', error);
        alert('Could not access webcam. Please grant permission.');
      }
    };

    setupCamera();

    // Cleanup function
    return () => {
      if (videoRef.current?.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      }
    };
  }, [model, isModelLoading]);

  // Perform object detection
  useEffect(() => {
    if (!model || !videoRef.current || !canvasRef.current) return;

    const detectObjects = async () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');

      const updateCanvas = () => {
        if (video.videoWidth && video.videoHeight) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
        }
      };

      video.addEventListener('loadedmetadata', updateCanvas);

      const detect = async () => {
        if (video.readyState === 4) {
          updateCanvas();
          
          const predictions = await model.detect(video);
          
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          
          predictions.forEach(prediction => {
            const [x, y, width, height] = prediction.bbox;
            
            // Draw bounding box
            ctx.strokeStyle = '#FF0000';
            ctx.lineWidth = 4;
            ctx.strokeRect(x, y, width, height);
            
            // Draw label background
            ctx.fillStyle = '#FF0000';
            const textWidth = ctx.measureText(prediction.class).width;
            ctx.fillRect(x, y - 30, textWidth + 10, 30);
            
            // Draw label text
            ctx.fillStyle = '#FFFFFF';
            ctx.font = '18px Arial';
            ctx.fillText(prediction.class, x + 5, y - 5);
          });

          setDetectedObjects(predictions);
        }
        requestAnimationFrame(detect);
      };

      detect();
    };

    detectObjects();
  }, [model]);

  return (
    <div className="flex flex-col items-center justify-center w-full h-full">
      <h1 className="text-3xl font-bold mb-4">Real-Time Object Detection</h1>
      
      {isModelLoading && <p className="text-lg">Loading model... Please wait.</p>}

      <div className="relative w-full max-w-2xl">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-auto rounded-lg shadow-lg"
          style={{ display: isModelLoading ? 'none' : 'block' }}
        />
        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0 w-full h-full"
          style={{ display: isModelLoading ? 'none' : 'block' }}
        />
      </div>

      <div className="mt-4">
        <h2 className="text-xl font-semibold">Detected Objects:</h2>
        <ul className="list-disc pl-5">
          {detectedObjects.map((obj, index) => (
            <li key={index}>
              {obj.class} ({Math.round(obj.score * 100)}% confidence)
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default ObjectDetector;