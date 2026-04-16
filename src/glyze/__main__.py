import subprocess
import sys
import importlib.util

def main():
    # Using sys.executable ensures it uses the same environment as your current script
    spec = importlib.util.find_spec("glyze")
    subprocess.run([sys.executable, "-m", "streamlit", "run", f"{spec.origin.removesuffix("__init__.py")}/ui/Home.py"])



if __name__ == "__main__":
    main()