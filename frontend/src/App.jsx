import React, { useState } from "react";
import "./App.css";

const CLASS_NAMES = [
  "Bacterial Pneumonia",
  "Corona Virus Disease",
  "Normal",
  "Tuberculosis",
  "Viral Pneumonia",
];

const STEPS = [
  { id: 1, label: "Patient Info" },
  { id: 2, label: "X-Ray Upload" },
  { id: 3, label: "Results" },
];

function getRecommendations(diagnosis, age) {
  const ageNum = Number(age);
  const isChild = !Number.isNaN(ageNum) && ageNum < 18;
  const isSenior = !Number.isNaN(ageNum) && ageNum >= 65;

  const common = [
    "This result is AI-generated and must be confirmed by a qualified clinician.",
    "Do not change or stop prescribed medication based on this screening alone.",
    "Seek urgent care if breathing becomes difficult, chest pain worsens, or oxygen levels drop.",
  ];

  const byDiagnosis = {
    Normal: [
      "No acute lung abnormality was suggested by the model for this X-ray.",
      "Continue routine health check-ups and maintain good respiratory hygiene.",
      "Return for reassessment if cough, fever, or shortness of breath develops.",
    ],
    "Bacterial Pneumonia": [
      "Bacterial pneumonia often requires clinical evaluation and may need antibiotic therapy.",
      "Monitor temperature, cough, sputum color, and hydration status closely.",
      "Rest adequately and avoid smoke or polluted environments during recovery.",
    ],
    "Viral Pneumonia": [
      "Viral pneumonia is usually managed with supportive care under medical supervision.",
      "Track fever, oxygen saturation, and breathing effort over the next 48–72 hours.",
      "Isolate as advised and follow local infection-control guidance if symptoms persist.",
    ],
    "Corona Virus Disease": [
      "COVID-19–related findings should be correlated with symptoms and PCR/antigen testing.",
      "Self-isolate and inform your healthcare provider about exposure and symptom timeline.",
      "Watch for warning signs: persistent high fever, breathlessness, confusion, or bluish lips.",
    ],
    Tuberculosis: [
      "Tuberculosis suspicion requires confirmatory tests such as sputum analysis and clinical workup.",
      "Avoid close contact with others until a doctor evaluates infectivity risk.",
      "Complete the full treatment course if TB is confirmed—partial treatment can be harmful.",
    ],
  };

  const ageNotes = [];
  if (isChild) {
    ageNotes.push(
      "Pediatric patients need age-appropriate dosing and closer monitoring by a pediatrician."
    );
  }
  if (isSenior) {
    ageNotes.push(
      "Older adults have higher complication risk—consider earlier in-person medical review."
    );
  }

  return [...(byDiagnosis[diagnosis] || byDiagnosis.Normal), ...ageNotes, ...common];
}

function App() {
  const [step, setStep] = useState(1);
  const [patientName, setPatientName] = useState("");
  const [patientAge, setPatientAge] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setSelectedFile(file);
    setResult(null);
    setError(null);

    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  };

  const handlePatientContinue = (e) => {
    e.preventDefault();
    setError(null);

    const trimmedName = patientName.trim();
    const age = Number(patientAge);

    if (!trimmedName) {
      setError("Please enter the patient's name.");
      return;
    }
    if (!patientAge || Number.isNaN(age) || age <= 0 || age > 120) {
      setError("Please enter a valid age between 1 and 120.");
      return;
    }

    setStep(2);
  };

  const handleRunDiagnostics = async () => {
    if (!selectedFile) {
      setError("Please upload an X-ray image first.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch("http://127.0.0.1:5000/api/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || "Failed to get prediction from server");
      }

      const data = await response.json();
      setResult(data);
      setStep(3);
    } catch (err) {
      console.error(err);
      setError(err.message || "Something went wrong during prediction.");
    } finally {
      setLoading(false);
    }
  };

  const handleStartOver = () => {
    setStep(1);
    setPatientName("");
    setPatientAge("");
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
  };

  const confidencePercent = result ? (result.confidence || 0) * 100 : 0;
  const recommendations = result
    ? getRecommendations(result.predicted_class, patientAge)
    : [];

  return (
    <div className="app-root">
      <header className="app-header">
        <h1>🫁 Pneumonia Detection AI</h1>
        <h2>Advanced Chest X-Ray Analysis System</h2>
      </header>

      <nav className="stepper" aria-label="Progress">
        {STEPS.map((item, index) => (
          <div
            key={item.id}
            className={`stepper-item ${step >= item.id ? "active" : ""} ${
              step === item.id ? "current" : ""
            }`}
          >
            <span className="stepper-circle">{item.id}</span>
            <span className="stepper-label">{item.label}</span>
            {index < STEPS.length - 1 && <span className="stepper-line" />}
          </div>
        ))}
      </nav>

      <main className="app-main app-main--single">
        {step === 1 && (
          <section className="panel panel--wide">
            <h3>Patient Information</h3>
            <p className="hint">
              Enter basic details before uploading the chest X-ray.
            </p>

            <form className="patient-form" onSubmit={handlePatientContinue}>
              <label className="field">
                <span>Patient Name</span>
                <input
                  type="text"
                  value={patientName}
                  onChange={(e) => setPatientName(e.target.value)}
                  placeholder="e.g. John Smith"
                  autoComplete="name"
                />
              </label>

              <label className="field">
                <span>Age</span>
                <input
                  type="number"
                  min="1"
                  max="120"
                  value={patientAge}
                  onChange={(e) => setPatientAge(e.target.value)}
                  placeholder="e.g. 45"
                />
              </label>

              {error && <p className="error">{error}</p>}

              <button type="submit" className="primary-btn">
                Continue to X-Ray Upload
              </button>
            </form>
          </section>
        )}

        {step === 2 && (
          <section className="panel panel--wide">
            <div className="patient-summary">
              <div>
                <span className="summary-label">Patient</span>
                <strong>{patientName}</strong>
              </div>
              <div>
                <span className="summary-label">Age</span>
                <strong>{patientAge} years</strong>
              </div>
              <button
                type="button"
                className="text-btn"
                onClick={() => {
                  setError(null);
                  setStep(1);
                }}
              >
                Edit details
              </button>
            </div>

            <h3>Upload Chest X-Ray</h3>
            <label className="upload-box">
              <input
                type="file"
                accept="image/jpeg,image/png"
                onChange={handleFileChange}
              />
              <span>Drop your image here or click to browse</span>
            </label>

            {previewUrl && (
              <div className="preview">
                <img src={previewUrl} alt="X-ray preview" />
              </div>
            )}

            {error && <p className="error">{error}</p>}

            <button
              className="primary-btn"
              onClick={handleRunDiagnostics}
              disabled={!selectedFile || loading}
            >
              {loading ? "Analyzing X-Ray..." : "Run Diagnostics"}
            </button>
          </section>
        )}

        {step === 3 && result && (
          <section className="panel panel--wide">
            <div className="patient-summary patient-summary--result">
              <div>
                <span className="summary-label">Patient</span>
                <strong>{patientName}</strong>
              </div>
              <div>
                <span className="summary-label">Age</span>
                <strong>{patientAge} years</strong>
              </div>
              <div>
                <span className="summary-label">Screening Date</span>
                <strong>{new Date().toLocaleDateString()}</strong>
              </div>
            </div>

            <div className="results-layout">
              <div className="results-card">
                <h3>Analysis Results</h3>

                <div className="metric">
                  <span className="metric-label">Diagnosis</span>
                  <span className="metric-value diagnosis-badge">
                    {result.predicted_class}
                  </span>
                </div>

                <div className="metric">
                  <span className="metric-label">Confidence</span>
                  <span className="metric-value">
                    {confidencePercent.toFixed(2)}%
                  </span>
                </div>

                <div className="bar-container">
                  <div
                    className="bar-fill"
                    style={{ width: `${confidencePercent}%` }}
                  />
                </div>

                <h4>Probability Distribution</h4>
                <div className="prob-chart">
                  {CLASS_NAMES.map((name) => {
                    const p =
                      result.probabilities && result.probabilities[name]
                        ? result.probabilities[name]
                        : 0;
                    const pPercent = p * 100;
                    return (
                      <div key={name} className="prob-row">
                        <span className="prob-label">{name}</span>
                        <div className="prob-bar-wrapper">
                          <div
                            className="prob-bar"
                            style={{ width: `${pPercent}%` }}
                          />
                        </div>
                        <span className="prob-value">
                          {pPercent.toFixed(1)}%
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>

              <div className="results-card recommendations-card">
                <h3>Recommendations</h3>
                <p className="hint">
                  Guidance based on the AI screening result for{" "}
                  <strong>{patientName}</strong>.
                </p>
                <ul className="recommendations-list">
                  {recommendations.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </div>
            </div>

            {previewUrl && (
              <div className="preview preview--compact">
                <img src={previewUrl} alt="Submitted X-ray" />
              </div>
            )}

            <button className="secondary-btn" onClick={handleStartOver}>
              Start New Screening
            </button>
          </section>
        )}
      </main>

      <footer className="app-footer">
        <hr />
        <p>
          <em>
            Note: This tool is for educational purposes only and should not be
            used for medical diagnosis.
          </em>
        </p>
      </footer>
    </div>
  );
}

export default App;
