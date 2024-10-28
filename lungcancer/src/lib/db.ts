import mongoose from 'mongoose';

const connectToDatabase = async () => {
  if (mongoose.connection.readyState >= 1) return; // Already connected

  await mongoose.connect(process.env.MONGODB_URI || '', {
    // Remove deprecated options
  });
};

export { connectToDatabase };
