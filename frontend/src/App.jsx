import { useState } from "react";

export default function LoanApprovalForm() {
    const featureNames = [
        "Credit Score",
        "Annual Income ($)",
        "Loan Amount ($)",
        "Loan Term (years)",
        "Employment Length (years)",
        "Debt-to-Income Ratio (%)",
        "Home Ownership (0=Rent, 1=Own, 2=Mortgage)",
        "Education Level (0=HS, 1=Bachelor, 2=Master, 3=PhD)",
        "Marital Status (0=Single, 1=Married, 2=Divorced)",
        "Age",
        "Employment Status (0=Unemployed, 1=Employed, 2=Self-Employed)",
        "Credit History (0=No, 1=Yes)",
        "Loan Purpose (0=Personal, 1=Business, 2=Education)",
        "Property Value ($)"
    ];
    
    const [features, setFeatures] = useState(Array(featureNames.length).fill(""));
    const [result, setResult] = useState(null);

    const handleChange = (index, value) => {
        const newFeatures = [...features];
        newFeatures[index] = value;
        setFeatures(newFeatures);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        try {
            const response = await fetch("http://127.0.0.1:8000/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ features: features.map(Number) }),
            });

            if (!response.ok) throw new Error("Failed to fetch prediction");

            const data = await response.json();
            setResult(data.prediction);
        } catch (error) {
            console.error("Error:", error);
            setResult("Error fetching prediction");
        }
    };

    return (
        <div className="p-8 max-w-xl mx-auto bg-gradient-to-br from-blue-50 to-white shadow-xl rounded-xl border border-gray-300">
            <h2 className="text-3xl font-extrabold mb-6 text-center text-blue-700">Loan Approval Prediction</h2>
            <form onSubmit={handleSubmit} className="space-y-4">
                {featureNames.map((name, index) => (
                    <div key={index} className="flex flex-col">
                        <label className="font-semibold text-gray-800 mb-1">{name}</label>
                        <input
                            type="number"
                            value={features[index]}
                            onChange={(e) => handleChange(index, e.target.value)}
                            placeholder={`Enter ${name}`}
                            className="w-full p-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 bg-gray-100 text-gray-800"
                            required
                        />
                    </div>
                ))}
                <button type="submit" className="w-full bg-blue-600 text-white p-3 rounded-lg hover:bg-blue-700 transition font-semibold shadow-md">Predict</button>
            </form>
            {result && <p className="mt-6 text-lg font-bold text-center text-blue-800">Prediction: {result}</p>}
        </div>
    );
}
