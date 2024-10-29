// src/ components/ PredictionForm.tsx

import { useState } from 'react';
import axios from 'axios';
import { useRouter } from 'next/router';

type FormData = {
  Age: string;
  AirPollution: string;
  AlcoholUse: string;
  DustAllergy: string;
  OccupationalHazards: string;
  GeneticRisk: string;
  ChronicLungDisease: string;
  BalancedDiet: string;
  Obesity: string;
  Smoking: string;
  PassiveSmoker: string;
  ChestPain: string;
  CoughingOfBlood: string;
  Fatigue: string;
  WeightLoss: string;
  ShortnessOfBreath: string;
  Wheezing: string;
  SwallowingDifficulty: string;
  ClubbingOfFingernails: string;
  FrequentCold: string;
  DryCough: string;
  Snoring: string;
};

const PredictForm = () => {
  const [formData, setFormData] = useState<FormData>({
    Age: '',
    AirPollution: '',
    AlcoholUse: '',
    DustAllergy: '',
    OccupationalHazards: '',
    GeneticRisk: '',
    ChronicLungDisease: '',
    BalancedDiet: '',
    Obesity: '',
    Smoking: '',
    PassiveSmoker: '',
    ChestPain: '',
    CoughingOfBlood: '',
    Fatigue: '',
    WeightLoss: '',
    ShortnessOfBreath: '',
    Wheezing: '',
    SwallowingDifficulty: '',
    ClubbingOfFingernails: '',
    FrequentCold: '',
    DryCough: '',
    Snoring: '',
  });

  const router = useRouter();

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prevData) => ({ ...prevData, [name]: value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const response = await axios.post('/api/predict', { inputs: formData });
      router.push({
        pathname: '/results',
        query: { prediction: response.data.prediction },
      });
    } catch (error) {
      console.error(error);
      alert('Prediction failed!');
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex flex-col max-w-md mx-auto mt-10">
      {Object.keys(formData).map((key) => (
        <label key={key}>
          {key}:
          <input
            type="number"
            name={key}
            value={formData[key as keyof FormData]} // Type assertion added
            onChange={handleChange}
            className="p-2 mb-4 border border-gray-300 rounded"
          />
        </label>
      ))}
      <button type="submit" className="bg-blue-600 text-white p-2 rounded">Predict</button>
    </form>
  );
};

export default PredictForm;
