// src/api/auth.ts

import type { NextApiRequest, NextApiResponse } from 'next';
import { connectToDatabase } from '../../lib/db';
import bcrypt from 'bcryptjs';
import User from '../../models/User';

const authHandler = async (req: NextApiRequest, res: NextApiResponse) => {
  try {
    await connectToDatabase();

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
      return res.status(405).json({ message: `Method ${req.method} Not Allowed` });
    }
  } catch (error) {
    console.error('Error in authHandler:', error);
    return res.status(500).json({ message: 'Internal Server Error' });
  }
};

export default authHandler;