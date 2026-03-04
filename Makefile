CXX ?= g++
CXXFLAGS ?= -O3 -DNDEBUG -std=c++20 -pthread

TARGET = build/bench
SRC    = src/bench.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

run: all
	./$(TARGET)

clean:
	rm -rf build
