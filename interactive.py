from model import Model


if __name__ == "__main__":
    model = Model()
    print("loaded.")

    while True:
        sentence = input("> ")
        predicted = model.predict(sentence)
        print(predicted)
