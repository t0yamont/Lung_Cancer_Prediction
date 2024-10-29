// src/ pages/ signup.tsx

import Navbar from '../components/Navbar';
import SignupForm from '../components/SignupForm';

export default function SignupPage() {
  return (
    <div className="flex items-center justify-center min-h-screen">
      <SignupForm />
    </div>
  );
}
