import os
import subprocess


def main():
    # Run train.py to train the model and save the metrics
    print("Running training script...")
    subprocess.run(["python", "../model/train.py"])

    # Execute results.ipynb to visualize the results
    print("Executing results notebook...")
    subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            "../results/results.ipynb",
        ]
    )

    print(
        "Project run completed. Model and metrics saved, and results notebook executed."
    )


if __name__ == "__main__":
    main()
