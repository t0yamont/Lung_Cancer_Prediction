import Link from 'next/link';

const Navbar = () => {
  return (
    <nav className="flex justify-between items-center p-4 bg-blue-600 text-white">
      <h1 className="text-xl">Lung Cancer Prediction App</h1>
      <div>
        <Link href="/">Home</Link>
        <Link href="/login" className="ml-4">Login</Link>
        <Link href="/signup" className="ml-4">Signup</Link>
      </div>
    </nav>
  );
};

export default Navbar;
