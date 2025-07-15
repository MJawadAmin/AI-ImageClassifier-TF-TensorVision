// src/app/page.js
import ObjectDetector from '@/components/ObjectDectator';
import ClientOnly from '@/components/ClientOnly';
import './globals.css';

export const metadata = {
  title: 'TensorFlow.js Object Detector',
  description: 'A Next.js app for real-time object detection using TensorFlow.js',
};

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-4 bg-gray-100">
      <ClientOnly fallback={<div className="text-lg">Loading Object Detector...</div>}>
        <ObjectDetector />
      </ClientOnly>
    </main>
  );
}