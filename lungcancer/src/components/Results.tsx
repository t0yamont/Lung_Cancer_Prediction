// src/ components/ Results.tsx

import { useRouter } from 'next/router';

const Results = () => {
  const router = useRouter();
  const { prediction } = router.query;

  return (
    <div className="max-w-md mx-auto mt-10">
      <h2 className="text-xl mb-4">Prediction Result</h2>
      <p>Your predicted risk of contracting lung cancer is: <strong>{prediction}</strong></p>
      {/* Add lifestyle tips based on prediction */}
      {prediction && (
        <div className="mt-4">
          <h3>Lifestyle Tips:</h3>
          <p>Here are some tips based on your prediction...</p>
        </div>
      )}
    </div>
  );
};

export default Results;
