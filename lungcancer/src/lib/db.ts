// src/ lib/ db.ts

import mongoose from 'mongoose';

const connectToDatabase = async (retries = 5, delay = 5000) => {
  if (mongoose.connection.readyState >= 1) return;

  while (retries) {
    try {
      await mongoose.connect(process.env.MONGODB_URI || '');
      console.log('Connected to MongoDB');
      return;
    } catch (error) {
      console.error('Failed to connect to MongoDB', error);
      retries -= 1;
      console.log(`Retries left: ${retries}`);
      if (!retries) throw new Error('MongoDB connection failed');
      await new Promise(res => setTimeout(res, delay));
    }
  }
};

export { connectToDatabase };