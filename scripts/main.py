from improve_email import improve_email
import classifier

def main():
    email = None  # TODO: pull email data from source
    passes = 0
    history = []

    while classifier.classify(email) == 1:
        passes += 1
        email = improve_email(email)
        history.append({"pass": passes, "email": email})

    print(f"Passes: {passes}")
    print(f"History: {history}")


if __name__ == "__main__":
    main()
