import { useState } from "react";
import "./App.css"; // Import external CSS file

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
        <div className="loan-container">
            <h2 className="loan-title">Loan Approval Prediction</h2>
            <form onSubmit={handleSubmit} className="loan-form">
                {featureNames.map((name, index) => (
                    <div key={index} className="loan-input-group">
                        <label>{name}</label>
                        <input
                            type="number"
                            value={features[index]}
                            onChange={(e) => handleChange(index, e.target.value)}
                            placeholder={`Enter ${name}`}
                            required
                        />
                    </div>
                ))}
                <button type="submit" className="loan-button">Predict</button>
            </form>
            {result && <p className="loan-result">Prediction: {result}</p>}
        </div>
    );
}
