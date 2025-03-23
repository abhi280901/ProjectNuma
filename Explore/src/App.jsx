import React, { useState } from "react";
import axios from "axios";
import { motion } from "framer-motion";
import 'bootstrap/dist/css/bootstrap.min.css';

function App() {
    const [file, setFile] = useState(null);
    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(false);
    const [pcaData, setPcaData] = useState({ ori_data: [], svd_noisy: [] });

    const handleFileChange = (e) => setFile(e.target.files[0]);

    const handleUpload = async () => {
        if (!file) return alert("Please select a file first!");
    
        const formData = new FormData();
        formData.append("file", file);
    
        try {
            setLoading(true);
            setPrediction(null);
    
            const response = await axios.post("http://localhost:8000/predict", formData);
            console.log("Full response data:", response.data);
    
            if (!response.data) {
                console.error("Response data is undefined:", response);
                alert("Response is invalid!");
                return;
            }
    
            const { classification_report, sample_predictions } = response.data;
    
            setPrediction({
                report: classification_report,
                results: sample_predictions,
            });
        } catch (error) {
            console.error("Error uploading file:", error);
            alert("Failed to get prediction.");
        } finally {
            setLoading(false);
        }
    };

    const handleRunPCA = async () => {
        if (!file) return alert("Please select a file first!");

        const formData = new FormData();
        formData.append("file", file);

        try {
            setLoading(true);
            const response = await axios.post("http://localhost:8000/run_pca", formData);
            console.log("PCA response data:", response.data);
            setPcaData(response.data);
            await handleUpload(); 
        } catch (error) {
            console.error("Error running PCA:", error);
            alert("Failed to run PCA.");
        } finally {
            setLoading(false);
        }
    };

    const predictions = prediction?.results || [];
    const report = prediction?.report || [];

    return (
        <div className="container-fluid bg-light vh-100 d-flex align-items-center justify-content-center">
            <div className="row w-75 shadow-lg rounded-3 bg-white p-4">
                
                {/* Left Side: Inputs and Buttons */}
                <div className="col-md-4 border-end">
                    <motion.h1
                        initial={{ y: -20, opacity: 0 }}
                        animate={{ y: 0, opacity: 1 }}
                        transition={{ duration: 0.5 }}
                        className="text-center mb-4 text-black"
                    >
                        FedShield
                    </motion.h1>
                    <input
                        type="file"
                        onChange={handleFileChange}
                        className="form-control mb-3"
                    />
                    <button onClick={handleUpload} className="btn btn-success w-100 mb-2">Upload & Predict</button>
                    <button onClick={handleRunPCA} className="btn btn-info w-100 mb-2">Run PCA</button>
                </div>

                {/* Right Side: Output */}
                <div className="col-md-8">
                    {loading && <div className="text-center text-warning">🔄 Processing...</div>}

                    {pcaData.ori_data.length > 0 && (
                        <div>
                            <h3 className="text-secondary">Original Data</h3>
                            <pre className="bg-light p-2 rounded border border-dark">{pcaData.ori_data.join('\n')}</pre>
                            <h3 className="text-secondary mt-3">PCA Data</h3>
                            <pre className="bg-light p-2 rounded border border-dark">{pcaData.svd_noisy.join('\n')}</pre>
                            <p className="text-muted fst-italic small mt-1">
                                Note: The PCA-ed table above has included Gaussian noise.
                            </p>
                        </div>
                    )}

                    {predictions.length > 0 && (
                        <div className="mt-4">
                            <h2 className="text-black">Prediction Results</h2>
                            <table className="table table-striped table-bordered">
                                <thead className="table-dark">
                                    <tr>
                                        <th>Transaction</th>
                                        <th>Prediction</th>
                                        <th>Labels</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {predictions.map((res, index) => {
                                        const [_, status, target] = res.split(":").map(item => item.trim());
                                        return (
                                            <tr key={index} className={status === "Fraud" ? "table-danger" : "table-success"}>
                                                <td>{index}</td>
                                                <td className="fw-bold">{status}</td>
                                                <td className="fw-bold">{target}</td>
                                            </tr>
                                        );
                                    })}
                                </tbody>
                            </table>
                            <p className="text-muted fst-italic small mt-1">
                                Model: Using Split Federated Learning framework on CTGAN (to handle imbalanced labels) and ResNet (for prediction).
                            </p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

export default App;