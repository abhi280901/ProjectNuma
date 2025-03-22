import React, { useState } from "react";
import axios from "axios";
import { motion } from "framer-motion";

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
            alert("PCA processing complete!");
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
        <div className="min-h-screen bg-gradient-to-r from-blue-500 to-purple-600 flex flex-col items-center justify-center p-4">
            <motion.h1
                initial={{ y: -20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ duration: 0.5 }}
                className="text-4xl font-extrabold mb-6 text-white drop-shadow-lg"
            >
                🌟 FedShield 🌟
            </motion.h1>

            <motion.div
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ duration: 0.5 }}
                className="bg-white p-6 rounded-2xl shadow-lg w-80"
            >
                <input
                    type="file"
                    onChange={handleFileChange}
                    className="w-full mb-4 p-2 border rounded"
                />
                <button
                    onClick={handleUpload}
                    className="w-full bg-gradient-to-r from-blue-500 to-purple-600 hover:opacity-90 text-white font-semibold py-2 rounded-2xl transition"
                >
                    🚀 Upload & Predict
                </button>
                <button
                    onClick={handleRunPCA}
                    className="w-full mt-2 bg-gradient-to-r from-green-500 to-teal-600 hover:opacity-90 text-white font-semibold py-2 rounded-2xl transition"
                >
                    🔍 Run PCA
                </button>

                {loading && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ duration: 0.3 }}
                        className="text-blue-500 font-semibold mt-4 text-center"
                    >
                        🔄 Processing...
                    </motion.div>
                )}

                {pcaData.ori_data.length > 0 && (
                    <div className="mt-4">
                        <h3 className="text-lg font-semibold mb-2">📌 Original Data</h3>
                        <pre>{pcaData.ori_data.join('\n')}</pre>
                        <h3 className="text-lg font-semibold mt-4 mb-2">📌 PCA Data</h3>
                        <pre>{pcaData.svd_noisy.join('\n')}</pre>
                    </div>
                )}

                {predictions.length > 0 && (
                    <motion.div
                        initial={{ scale: 0.8, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        transition={{ duration: 0.5 }}
                        className="mt-4 p-4 bg-gray-100 text-gray-800 rounded-xl shadow-lg text-center"
                    >
                        <h2 className="text-2xl font-bold mb-4 text-green-700">✅ Prediction Results</h2>
                        <table className="table-auto border-collapse border border-gray-400 w-full">
                            <thead>
                                <tr className="bg-gray-200">
                                    <th className="border border-gray-400 px-4 py-2">Transaction</th>
                                    <th className="border border-gray-400 px-4 py-2">Prediction</th>
                                    <th className="border border-gray-400 px-4 py-2">Labels</th>
                                </tr>
                            </thead>
                            <tbody>
                                {predictions.map((res, index) => {
                                    const [_, status, target] = res.split(":").map(item => item.trim());
                                    return (
                                        <tr key={index} className={status === "Fraud" ? "bg-red-100" : "bg-green-100"}>
                                            <td className="border border-gray-400 px-4 py-2">{`${index}`}</td>
                                            <td className="border border-gray-400 px-4 py-2 font-bold">{status}</td>
                                            <td className="border border-gray-400 px-4 py-2 font-bold">{target}</td>
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    </motion.div>
                )}
            </motion.div>
        </div>
    );
}

export default App;
