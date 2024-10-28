import Link from 'next/link';

export default function Home() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen">
      <h1 className="text-4xl">Lung Cancer Prediction</h1>
      <Link href="/login">
        <a className="mt-4 p-2 bg-blue-500 text-white rounded">Login</a>
      </Link>
      <Link href="/signup">
        <a className="mt-4 p-2 bg-blue-500 text-white rounded">Sign Up</a>
      </Link>
      <Link href="/predict">
        <a className="mt-4 p-2 bg-green-500 text-white rounded">Predict</a>
      </Link>
    </div>
  );
}
