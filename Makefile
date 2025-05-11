CXX = g++

GTEST_FLAGS = -lgtest -lgtest_main -pthread 

SRC = etc.cpp tensor.cpp op.cpp nn.cpp
TEST_SRC = etc_tests.cpp tensor.cpp op.cpp nn.cpp

MAIN = etc
TEST = etc_test

all: $(MAIN) $(TEST)

$(MAIN): $(SRC)
	$(CXX) $(SRC) -o $(MAIN)
	
$(TEST): $(TEST_SRC)
	$(CXX) $(TEST_SRC) $(GTEST_FLAGS) -o $(TEST)
	
test: $(TEST)
	./$(TEST)
	
clean: 
	rm -f $(MAIN) $(TEST)