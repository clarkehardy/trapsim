CXX      = clang++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra
TARGET   = laplace

# Optional OpenMP (requires: brew install libomp)
# Uncomment the two lines below to enable parallel red-black sweeps.
# CXXFLAGS += -Xpreprocessor -fopenmp
# LDFLAGS  += -lomp

$(TARGET): laplace.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) laplace.cpp $(LDFLAGS)

clean:
	rm -f $(TARGET)
