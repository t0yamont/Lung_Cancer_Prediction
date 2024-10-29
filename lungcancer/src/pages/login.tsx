// src/ pages/ login.tsx

import Navbar from '../components/Navbar';
import LoginForm from '../components/LoginForm';

export default function LoginPage() {
  return (
    <div className="flex items-center justify-center min-h-screen">
      <LoginForm />
    </div>
  );
}
