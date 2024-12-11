#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

// Function Declarations (Prototypes)
void printMessage(std::string message = "Default message");
inline int multiply(int a, int b);
void reverseString(std::string s);
bool isEven(int number);
void printEvenNumbers(const std::vector<int>& numbers);
void lambdaExample();

class Animal {
 public:
  // Constructor
  Animal(std::string name) : name(name) {}

  // Virtual Function (to demonstrate polymorphism)
  virtual void makeSound() {
    std::cout << name << " makes a sound." << std::endl;
  }

  // Member Function
  void sleep() {
    std::cout << name << " is sleeping." << std::endl;
  }

 protected:
  std::string name;
};

// Derived Class (Inheritance)
class Dog : public Animal {
 public:
  // Constructor using Base Class constructor
  Dog(std::string name, std::string breed) : Animal(name), breed(breed) {}

  // Overriding Virtual Function
  void makeSound() override {
    std::cout << name << " barks!" << std::endl;
  }

  // Static Function
  static void staticFunction() {
    std::cout << "This is a static function." << std::endl;
  }

  // Constant Member Function
  void describe() const {
    std::cout << name << " is a " << breed << "." << std::endl;
  }

 private:
  std::string breed;
};

// Another Derived Class for Polymorphism Demo
class Cat : public Animal {
 public:
  Cat(std::string name) : Animal(name) {}

  void makeSound() override {
    std::cout << name << " meows!" << std::endl;
  }
};

// Templated Function
template <typename T>
T max(T a, T b) {
  return (a > b) ? a : b;
}

int main() {
  // Call to Function with Default Argument
  printMessage("Custom message!");

  // Inline Function
  std::cout << "Multiply: " << multiply(6, 7) << std::endl;

  // Object Creation and Function Calls
  Dog dog("Buddy", "Golden Retriever");
  dog.makeSound();  // Polymorphism
  dog.sleep();
  dog.describe();

  Cat cat("Whiskers");
  cat.makeSound();  // Polymorphism
  cat.sleep();

  // Static Function Call
  Dog::staticFunction();

  // Templated Function Call
  std::cout << "Max of 3 and 7: " << max(3, 7) << std::endl;
  std::cout << "Max of 3.14 and 2.71: " << max(3.14, 2.71) << std::endl;

  // Lambda Expression Example
  lambdaExample();

  // Calling newly defined functions
  reverseString("Hello, World!");  // Reverse string
  std::cout << "Is 4 even? " << (isEven(4) ? "Yes" : "No") << std::endl;

  std::vector<int> numbers = {1, 2, 3, 4, 5, 6};
  printEvenNumbers(numbers);  // Print even numbers from the vector

  return 0;
}

// Function Implementations (after main function)

// Function with Default Arguments Implementation
void printMessage(std::string message) {
  std::cout << message << std::endl;
}

// Inline Function Implementation
inline int multiply(int a, int b) {
  return a * b;
}

// Function to reverse a string
void reverseString(std::string s) {
  std::reverse(s.begin(), s.end());
  std::cout << "Reversed string: " << s << std::endl;
}

// Function to check if a number is even
bool isEven(int number) {
  return (number % 2 == 0);
}

// Function to print all even numbers in a vector
void printEvenNumbers(const std::vector<int>& numbers) {
  std::cout << "Even numbers: ";
  for (int num : numbers) {
    if (isEven(num)) {
      std::cout << num << " ";
    }
  }
  std::cout << std::endl;
}

// Lambda Expression Example
void lambdaExample() {
  auto lambda = [](int x, int y) -> int { return x + y; };
  std::cout << "Lambda result: " << lambda(5, 7) << std::endl;
}
