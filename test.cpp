#include <gtest/gtest.h>
#include <iostream>

#include "kernels.h"

TEST(Tests, Test1) {
  std::cerr << "test" << std::endl;
  ASSERT_EQ(0, 1);
}
