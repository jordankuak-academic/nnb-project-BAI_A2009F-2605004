from app.services.preprocessor import Preprocessor


def main() -> None:
    preprocessor = Preprocessor()
    preprocessor.run()


if __name__ == "__main__":
    main()
