import argparse
import sys
import os
import subprocess
from colorama import Fore, Style

def run_app():
    print(f"{Fore.CYAN}Launching Streamlit app...{Style.RESET_ALL}")
    subprocess.run(["streamlit", "run", "src/survey/survey.py"], cwd=os.path.dirname(__file__))

def main():
    parser = argparse.ArgumentParser(description="Stock Prediction Model CLI")
    parser.add_argument('command', choices=['app'], help="Command to execute")
    args = parser.parse_args()

    if args.command == 'app':
        run_app()

if __name__ == "__main__":
    main()