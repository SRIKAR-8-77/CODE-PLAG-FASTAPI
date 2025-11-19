# Run with: uvicorn app:app --reload

# This file is for launching the FastAPI server for the PlagiarismCheckSystem
# You can use this file to start the server from the command line

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)