#include <iostream>
#include <stdexcept>

// Global function declaration (definition below)
void globalFunction();

// Class with member functions
class MyClass {
 public:
  // Constructor
  MyClass(int value) : value(value) {}

  // Member function defined inside the class
  void memberFunction() {
    if (value > 0) {  // If-else block inside a function
      std::cout << "Value is positive" << std::endl;
    } else if (value == 0) {
      std::cout << "Value is zero" << std::endl;
    } else {
      std::cout << "Value is negative" << std::endl;
    }
  }

  // Member function declared inside the class, defined outside
  void anotherMemberFunction(int x);

 private:
  int value;
};

// Definition of anotherMemberFunction (outside the class)
void MyClass::anotherMemberFunction(int x) {
  for (int i = 0; i < x; ++i) {  // For loop
    std::cout << "i: " << i << std::endl;
  }

  int count = 0;
  while (count < x) {  // While loop
    std::cout << "Count: " << count << std::endl;
    ++count;
  }
}

// Function with try-catch block
void tryCatchExample() {
  try {
    throw std::runtime_error("An error occurred");
  } catch (const std::exception& e) {  // Catch block
    std::cout << "Caught exception: " << e.what() << std::endl;
  }
}

// Function definition (declared above)
void globalFunction() {
  std::cout << "This is a global function" << std::endl;
}

int main() {
  // Calling global function
  globalFunction();

  // Creating an instance of MyClass
  MyClass obj(10);
  obj.memberFunction();  // Call member function

  // Calling a member function defined outside the class
  obj.anotherMemberFunction(3);

  // For loop with multiple variables
  for (int i = 0, j = 10; i < j; ++i, --j) {
    std::cout << "i: " << i << ", j: " << j << std::endl;
  }

  // Do-while loop
  int k = 0;
  do {
    std::cout << "k: " << k << std::endl;
    ++k;
  } while (k < 5);

  // Switch statement
  int choice = 2;
  switch (choice) {
    case 1:
      std::cout << "Choice is 1" << std::endl;
      break;
    case 2:
      std::cout << "Choice is 2" << std::endl;
      break;
    default:
      std::cout << "Unknown choice" << std::endl;
      break;
  }

  // Try-catch block
  tryCatchExample();

  return 0;
}
