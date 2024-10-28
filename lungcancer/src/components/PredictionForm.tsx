import { useState } from 'react';
import axios from 'axios';
import { useRouter } from 'next/router';

const PredictForm = () => {
  const [formData, setFormData] = useState({
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
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const response = await axios.post('/api/predict', { inputs: formData });
      router.push({
        pathname: '/results',
        query: { prediction: response.data.prediction }
      });
    } catch (error) {
      console.error(error);
      alert('Prediction failed!');
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex flex-col max-w-md mx-auto mt-10">
      <label>
        Age:
        <input type="number" name="Age" onChange={handleChange} className="p-2 mb-4 border border-gray-300 rounded" />
      </label>
      {Object.keys(formData).filter(key => key !== 'Age').map((key) => (
        <label key={key}>
          {key} (0-8):
          <input type="number" name={key} onChange={handleChange} max={8} className="p-2 mb-4 border border-gray-300 rounded" />
        </label>
      ))}
      <button type="submit" className="bg-blue-600 text-white p-2 rounded">Predict</button>
    </form>
  );
};

export default PredictForm;
