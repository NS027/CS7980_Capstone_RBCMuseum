# RAG Evaluation System

This project implements a Retrieval-Augmented Generation (RAG) evaluation system with a React frontend and FastAPI backend.

## Setup

1. Clone the repository
2. Set up the frontend:
   - Navigate to the `frontend` directory
   - Run `npm install`
   - Start the development server with `npm start`
3. Set up the backend:
   - Navigate to the `backend` directory
   - Create a virtual environment: `python -m venv venv`
   - Activate the virtual environment
   - Install dependencies: `pip install -r requirements.txt`
   - Set your OpenAI API key in the `.env` file
   - Start the FastAPI server: `uvicorn app.main:app --reload`

## Usage

1. Open the frontend in your browser (usually at `http://localhost:3000`)
2. Select the desired LLMs and test size
3. Click "Evaluate" to run the RAG evaluation
4. View the results in the chart and detailed breakdown

## Testing

Run backend tests using pytest: `pytest backend/tests`