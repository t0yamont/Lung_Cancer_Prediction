// src/api/predict.ts

import type { NextApiRequest, NextApiResponse } from 'next';
import { spawn } from 'child_process';

const predictHandler = async (req: NextApiRequest, res: NextApiResponse) => {
  if (req.method === 'POST') {
    const { inputs } = req.body;

    // Ensure the path to your Python script is correct
    const pythonProcess = spawn('python', ['src/utils/predictor.py', JSON.stringify(inputs)]);

    let prediction = '';
    let errorMessage = '';

    pythonProcess.stdout.on('data', (data) => {
      prediction += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      errorMessage += data.toString();
      console.error(`Python Error: ${data.toString()}`);
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        res.status(200).json({ prediction: prediction.trim() });
      } else {
        console.error(`Python process exited with code ${code}`);
        res.status(500).json({ error: errorMessage || 'Prediction process failed' });
      }
    });
  } else {
    res.setHeader('Allow', ['POST']);
    res.status(405).json({ message: `Method ${req.method} Not Allowed` });
  }
};

export default predictHandler;
