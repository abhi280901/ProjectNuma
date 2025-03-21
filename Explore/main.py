from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import os

app = FastAPI()

# CORS setup to allow frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        
        # Define base directory and ensure paths are correct
        base_dir = os.path.dirname(__file__)
        script_path = os.path.join(base_dir, "FinalImplementation", "SplitFed_ResNet_CTGAN_BIS_Demo.py")
        data_path = os.path.join(base_dir, "FinalImplementation", "demo_data.csv")
        model_path = os.path.join(base_dir, "FinalImplementation", "best_client_model_CTGAN_ResNet_V2_wo_drop_128_005.pth")

        # Check if model file exists
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        # Save the uploaded file in the correct location
        with open(data_path, "wb") as f:
            f.write(await file.read())

        # Run the model script with the corrected path
        result = subprocess.run(["python3", script_path], capture_output=True, text=True)

        print("STDOUT:", result.stdout)  # Debug output
        print("STDERR:", result.stderr)    # Debug errors

        # Extract prediction from output
        output_lines = result.stdout.splitlines()
        # Parse accuracy and classification report
        accuracy = None
        classification_report = []
        predictions = []

        for line in output_lines:
            if "Accuracy on new data:" in line:
                accuracy = line.strip()
            elif "Classification Report:" in line:
                classification_report = output_lines[output_lines.index(line) + 1 : output_lines.index(line) + 6]
            elif "Predictions:" in line:
                predictions = output_lines[output_lines.index(line) + 1 : output_lines.index(line) + 6]

        if not accuracy or not predictions:
            raise HTTPException(status_code=500, detail="Failed to retrieve predictions")

        # Return structured output
        return JSONResponse(content={
            "classification_report": classification_report,
            "sample_predictions": predictions,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")