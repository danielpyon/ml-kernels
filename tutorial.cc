#include <iostream>

#include "TutorialConfig.h"
#include "kernels.h"

int main(int argc, char* argv[]) {
  std::cout << "Version: " << kernels_VERSION_MAJOR << std::endl;
  std::cout << kernels::test() << std::endl;
  return 0;
}
