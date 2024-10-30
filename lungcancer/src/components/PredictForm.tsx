// src/components/PredictionForm.tsx

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
    if (name !== 'Age') {
      const numericValue = parseInt(value, 10);
      if (numericValue >= 0 && numericValue <= 8) {
        setFormData({
          ...formData,
          [name]: value,
        });
      }
    } else {
      setFormData({
        ...formData,
        [name]: value,
      });
    }
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
    <form onSubmit={handleSubmit} className="flex flex-col max-w-md mx-auto mt-10 space-y-6">
      <h2 className="text-xl font-bold mb-4 text-center">Lung Cancer Risk Prediction</h2>
      
      <label className="flex flex-col mb-2">
        Age:
        <input
          type="number"
          name="Age"
          value={formData.Age}
          onChange={handleChange}
          className="w-full p-2 mt-1 border border-gray-300 rounded"
        />
      </label>
      
      <h3 className="text-lg font-semibold mt-6 mb-4">Health Factors</h3>

      {Object.keys(formData).filter((key) => key !== 'Age').map((key) => (
        <label key={key} className="flex flex-col mb-2">
          {key.replace(/([A-Z])/g, ' $1').trim()}:
          <input
            type="number"
            name={key}
            value={formData[key as keyof FormData]}
            onChange={handleChange}
            className="w-full p-2 mt-1 border border-gray-300 rounded"
          />
        </label>
      ))}

      <button type="submit" className="w-full bg-blue-600 text-white p-3 rounded mt-6">Predict</button>
    </form>
  );
};

export default PredictForm;
