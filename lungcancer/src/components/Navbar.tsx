// src/ components/ Navbar.tsx

import Link from 'next/link';

const Navbar = () => {
  return (
    <nav className="flex justify-between items-center p-4 bg-blue-600 text-white">
      <h1 className="text-xl">Lung Cancer Prediction App</h1>
      <div>
        <Link href="/" legacyBehavior><a>Home</a></Link>
        <Link href="/login" legacyBehavior><a className="ml-4">Login</a></Link>
        <Link href="/signup" legacyBehavior><a className="ml-4">Signup</a></Link>
      </div>
    </nav>
  );
};

export default Navbar;