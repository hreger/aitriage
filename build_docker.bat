@echo off
echo Building ED-AI Triage System Docker image...
docker-compose build

echo Starting ED-AI Triage System...
docker-compose up -d

echo.
echo ED-AI Triage System is running!
echo Streamlit app: http://localhost:8501
echo Jupyter (if enabled): http://localhost:8888
echo.
echo To stop: docker-compose down
echo To view logs: docker-compose logs -f
pause
