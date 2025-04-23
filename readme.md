# Topic Modeling Application

A Shiny web application for topic modeling and visualization of text data.

## Requirements

- Python 3.8 or higher
- Internet connection for initial setup

## Quick Start

### Windows

1. Double-click `setup.bat` to create a virtual environment and install dependencies
   - This can take a while (especially when it gets to "Installing collected packages"), it will notify you in the terminal when it's  complete
2. Double-click `run.bat` to start the application
3. Open your browser to http://localhost:8000

## Features

- Upload and process CSV data
- Extract and clean text content
- Generate topic models using BERTopic
- Visualize topics with interactive plots
- Export results as reports and visualizations

## Folder Structure

- `app_files/`: Application source code
  - `app.py`: Main application file
  - `modules/`: Core application modules
  - `www/`: Static web assets
- `outputs/`: Generated output files (created on first run)

## Troubleshooting

If you encounter any issues:

1. Make sure Python 3.8+ is installed and in your PATH
2. Check that you have activated the virtual environment
3. Verify all dependencies are installed with `pip list`
