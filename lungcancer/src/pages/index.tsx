// src/ pages/ index.tsx

import Link from 'next/link';

export default function Home() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen">
      <h1 className="text-4xl">Lung Cancer Prediction</h1>
      <Link href="/login" className="mt-4 p-2 bg-blue-500 text-white rounded">Login</Link>

      <Link href="/signup" className="mt-4 p-2 bg-blue-500 text-white rounded">Sign Up</Link>
      
      <Link href="/predict"className="mt-4 p-2 bg-green-500 text-white rounded">Predict</Link>
      
    </div>
  );
}
