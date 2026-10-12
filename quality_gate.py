
import json
import sys

MINIMUM_ACCURACY = 0.99

try:
    with open("metrics.json", "r") as file:
        metrics = json.load(file)

    accuracy = metrics["accuracy"]

    print(f"Model accuracy: {accuracy:.4f}")
    print(f"Minimum required: {MINIMUM_ACCURACY:.2f}")

    if accuracy >= MINIMUM_ACCURACY:
        print("QUALITY GATE PASSED")
    else:
        print("QUALITY GATE FAILED")
        sys.exit(1)

except (FileNotFoundError, KeyError, ValueError) as error:
    print(f"Quality gate error: {error}")
    sys.exit(1)
