import type { NextApiRequest, NextApiResponse } from 'next';
import { connectToDatabase } from '../lib/db';
import bcrypt from 'bcryptjs';
import User from '../models/User';

const authHandler = async (req: NextApiRequest, res: NextApiResponse) => {
  await connectToDatabase(); // Ensure database connection

  if (req.method === 'POST') {
    const { email, password } = req.body;

    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return res.status(409).json({ message: 'User already exists' });
    }

    const hashedPassword = await bcrypt.hash(password, 10);
    const newUser = new User({ email, password: hashedPassword });
    await newUser.save();
    return res.status(201).json({ message: 'User created successfully' });
  } else {
    res.setHeader('Allow', ['POST']);
    res.status(405).end(`Method ${req.method} Not Allowed`);
  }
};

export default authHandler;
