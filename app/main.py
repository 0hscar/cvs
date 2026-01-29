import sys


def main():
    print("Select use case:")
    print("1. Hand Gesture Recognition")
    choice = input("Enter choice (1): ")

    if choice == "1":
        from handRecognition.handRecognition import handRecognition

        handRecognition()
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)


if __name__ == "__main__":
    main()
