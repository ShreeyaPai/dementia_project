import React, { useState } from "react";
import "./App.css";

const App = () => {
  const [formData, setFormData] = useState({
    gender: "M",
    age: "",
    EDUC: "",
    SES: "",
    MMSE: "",
    CDR: "",
    eTIV: "",
    nWBV: "",
    ASF: ""
  });

  const [prediction, setPrediction] = useState("");

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const response = await fetch("http://127.0.0.1:8000/predict/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error("Failed to fetch prediction");
      }

      const data = await response.json();
      setPrediction(data.ANN_Prediction);
    } catch (error) {
      console.error("Error:", error);
      setPrediction("Error fetching prediction");
    }
  };

  return (
    <div className="background">
      <h2 className="page-title">
    <span className="page-title">Dementia</span> 
    <span className="page-title-2"> Detection</span>
</h2>
      <div className="container">
        <form onSubmit={handleSubmit} className="medical-form">
          
          <div className="form-group full-width">
            <label>Gender</label>
            <select name="gender" value={formData.gender} onChange={handleChange}>
              <option value="M">Male</option>
              <option value="F">Female</option>
            </select>
          </div>

          <div className="form-group">
            <label>Age</label>
            <input type="number" name="age" value={formData.age} onChange={handleChange} required />
          </div>

          <div className="form-group">
            <label>EDUC</label>
            <input type="number" name="EDUC" value={formData.EDUC} onChange={handleChange} required />
          </div>

          <div className="form-group">
            <label>SES</label>
            <input type="number" name="SES" value={formData.SES} onChange={handleChange} required />
          </div>

          <div className="form-group">
            <label>MMSE</label>
            <input type="number" name="MMSE" value={formData.MMSE} onChange={handleChange} required />
          </div>

          <div className="form-group">
            <label>CDR</label>
            <input type="number" name="CDR" value={formData.CDR} onChange={handleChange} required />
          </div>

          <div className="form-group">
            <label>eTIV</label>
            <input type="number" name="eTIV" value={formData.eTIV} onChange={handleChange} required />
          </div>

          <div className="form-group">
            <label>nWBV</label>
            <input type="number" name="nWBV" value={formData.nWBV} onChange={handleChange} required />
          </div>

          <div className="form-group">
            <label>ASF</label>
            <input type="number" name="ASF" value={formData.ASF} onChange={handleChange} required />
          </div>

          <button type="submit" className="submit-btn">Detect</button>
        </form>

        {prediction && (
          <h3 className={`prediction-result ${prediction === "Non-Demented" ? "non-demented" : "demented"}`}>
            {prediction}
          </h3>
        )}
      </div>
    </div>
  );
};

export default App;
