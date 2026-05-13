import os
from pathlib import Path
from flask import Flask, request, render_template, send_file
from train_model import train_model as _train, test_model_inference as _inference

FILE_PATH = Path("artifacts").resolve()
MODEL_PATH = Path("artifacts/model.pkl")
BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)

if not MODEL_PATH.exists():
    _train()
@app.route("/")
def index():
    return render_template("index.html", message=None, file_ready=False)

@app.route("/upload", methods=["POST"])
def test_and_inference():
    try:
        file = request.files["file"]
        print(f"The test filename passed is {file.filename}")
        if file.filename == "":
            return render_template("index.html", message="No file uploaded", file_ready=False)

        # Test and calculate the inference
        output_filename = _inference(file.filename)
        output_filename = output_filename.split("/")[-1]
        print(output_filename)

        return render_template(
            "index.html",
            message="Inference completed successfully!",
            file_ready=True,
            download_file=output_filename
        )
    except Exception as e:
        return render_template(
            "index.html",
            message=f"Error processing file: {str(e)}",
            file_ready=False
        )

@app.route("/download/<filename>")
def download_file(filename):
    file_path = BASE_DIR / "artifacts" / filename

    if not file_path.exists():
        return "File not found", 404
    
    return send_file(file_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
