import sys


def _import_hand_recognition():
    """
    Robust import helper that tries a few import locations so `main.py` can be
    executed from different working directories / import contexts.
    """
    # Preferred direct package import (when the package is on sys.path)
    try:
        from handRecognition.handRecognition import handRecognition

        return handRecognition
    except Exception:
        pass

    # Alternate import path when running as `app` package
    try:
        from app.handRecognition.handRecognition import handRecognition

        return handRecognition
    except Exception:
        pass

    # Try dynamic import as a last resort
    try:
        import importlib

        mod = importlib.import_module("handRecognition.handRecognition")
        return getattr(mod, "handRecognition")
    except Exception:
        pass

    raise ImportError(
        "Could not import handRecognition.handRecognition. "
        "Ensure the `app` package is on PYTHONPATH and the module exists."
    )


def main():
    print("Select use case:")
    print("1. Hand Gesture Recognition")

    # Allow command-line override: `python main.py 1`
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        try:
            choice = input("Enter choice (1): ")
        except Exception:
            # In non-interactive environments, default to the single available choice.
            choice = ""

    # Default to option 1 when input is empty
    if not choice or choice.strip() == "":
        choice = "1"

    if choice == "1":
        try:
            handRecognition = _import_hand_recognition()
        except ImportError as e:
            print(f"Failed to import handRecognition: {e}")
            sys.exit(1)

        try:
            handRecognition()
        except Exception as e:
            print(f"handRecognition failed: {e}")
            sys.exit(1)
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)


if __name__ == "__main__":
    main()
