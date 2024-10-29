// src/ components/ LoginForm.tsx

import { useState } from 'react';
import axios from 'axios';
import { useRouter } from 'next/router';

const LoginForm = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await axios.post('/api/auth', { email, password });
      router.push('/'); // Redirect to homepage on success
    } catch (error) {
      console.error(error);
      alert('Login failed!');
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex flex-col max-w-md mx-auto mt-10">
      <input
        type="email"
        placeholder="Email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        className="p-2 mb-4 border border-gray-300 rounded"
      />
      <input
        type="password"
        placeholder="Password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        className="p-2 mb-4 border border-gray-300 rounded"
      />
      <button type="submit" className="bg-blue-600 text-white p-2 rounded">Login</button>
    </form>
  );
};

export default LoginForm;
