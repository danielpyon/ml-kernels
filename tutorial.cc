#include <iostream>

#include "TutorialConfig.h"
#include "MathFunctions.h"

int main(int argc, char* argv[]) {
  std::cout << "Version: " << kernels_VERSION_MAJOR << std::endl;
  std::cout << mathfunctions::sqrt(9.0) << std::endl;
  return 0;
}
