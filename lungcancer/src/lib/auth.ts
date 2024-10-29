// src/ lib/ auth.ts

import bcrypt from 'bcryptjs';
import User from '../models/User';

export const verifyPassword = async (password: string, hashedPassword: string) => {
  return await bcrypt.compare(password, hashedPassword);
};
