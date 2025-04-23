import os, sys 
# Add app_files to Python path 
app_files_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_files") 
sys.path.insert(0, app_files_dir) 
 
# Import and run the app 
from app_files.app import app 
from shiny import run_app 
 
# Disable warnings 
import warnings 
warnings.filterwarnings("ignore", module="watchfiles") 
 
# Launch the app 
run_app(app, reload=False, launch_browser=True, port=8000) 
