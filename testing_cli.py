import argparse
import sys
import os
import subprocess
from colorama import Fore, Style
from src.survey.survey_question_optimizer import test_all_parameterizations

def run_app():
    print(f"{Fore.CYAN}Launching Streamlit app...{Style.RESET_ALL}")
    subprocess.run(["streamlit", "run", "src/survey/survey.py"], cwd=os.path.dirname(__file__))

def run_tests():
    print(f"{Fore.GREEN}Running tests...{Style.RESET_ALL}")
    test_all_parameterizations()

def main():
    parser = argparse.ArgumentParser(description="ETSA CLI")
    parser.add_argument('command', choices=['app', 'tests', 'ttw'], help="Command to execute")
    args = parser.parse_args()

    if args.command == 'app':
        run_app()
    elif args.command == 'tests':
        run_tests()
    elif args.command == 'ttw':
        print(f"{Fore.YELLOW}Running ttw.py...{Style.RESET_ALL}")
        subprocess.run(["python", "src/sentiment/ttw.py"], cwd=os.path.dirname(__file__))

if __name__ == "__main__":
    main()