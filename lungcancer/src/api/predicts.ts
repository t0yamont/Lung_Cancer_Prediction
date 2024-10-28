import type { NextApiRequest, NextApiResponse } from 'next';
import { spawn } from 'child_process';
import { connectToDatabase } from '../lib/db';

const predictHandler = async (req: NextApiRequest, res: NextApiResponse) => {
  await connectToDatabase(); // Ensure database connection

  if (req.method === 'POST') {
    const { inputs } = req.body;

    const pythonProcess = spawn('python', ['src/utils/predictor.py', JSON.stringify(inputs)]);
    
    pythonProcess.stdout.on('data', (data) => {
      const prediction = data.toString();
      res.status(200).json({ prediction });
    });
    
    pythonProcess.stderr.on('data', (data) => {
      console.error(`Error: ${data}`);
      res.status(500).json({ error: 'Prediction error' });
    });
  } else {
    res.setHeader('Allow', ['POST']);
    res.status(405).end(`Method ${req.method} Not Allowed`);
  }
};

export default predictHandler;
