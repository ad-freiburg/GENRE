from model import Model


if __name__ == "__main__":
    model = Model()
    print("loaded.")

    while True:
        text = input("> ")
        predicted = model.predict_paragraph(text)
        print(predicted)
