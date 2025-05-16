import random

class Person:
    def __init__(self, joined_protest=0, threshold=0.50):
        self.joined_protest = joined_protest
        self.threshold = threshold

    def decide(self, percent_protesting):
        if percent_protesting >= self.threshold:
            self.joined_protest = 1

class Protest:
    def __init__(self):
        self.num_citizens = 10000
        self.num_protesters = 0
        self.current_protesting = 0.0
        self.initial_protesting = 0.035

        self.persons = []
        self.random = random.Random()

        for _ in range(self.num_citizens):
            has_joined = 0
            chance = self.random.random()
            threshold = 0.0
            if chance < self.initial_protesting:
                threshold = 0.0
                has_joined = 1
            else:
                threshold = self.random.random()
                has_joined = 0
            person = Person(has_joined, threshold)
            self.persons.append(person)

        self.tabulate()

    def act(self):
        for person in self.persons:
            person.decide(self.current_protesting)

    def tabulate(self):
        self.num_protesters = sum(person.joined_protest for person in self.persons)
        self.current_protesting = self.num_protesters / self.num_citizens

def main():
    protest = Protest()
    for j in range(10, 101, 10):
        for _ in range(j):
            protest.act()
            protest.tabulate()
        print(f"t={j:4d} num protesters = {protest.num_protesters:4d}, percent = {protest.current_protesting:8.2f}")

if __name__ == "__main__":
    main()
