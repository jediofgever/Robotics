OMPL_DIR = /usr
CXXFLAGS = -g -std=c++11 # change to -g when debugging code
INCLUDE_FLAGS = -I${OMPL_DIR}/include
LD_FLAGS = -L${OMPL_DIR}/lib -lompl -lompl_app -lboost_program_options -lboost_system
CXX=g++

MyRigidBodyPlanning: MyRigidBodyPlanning.o randomtree.o
	$(CXX) $(CXXFLAGS) -o MyRigidBodyPlanning MyRigidBodyPlanning.o randomtree.o $(LD_FLAGS)

clean:
	rm -rf *.o
	rm -rf *.log
	rm -rf *.db
	rm -rf *.console
	rm -rf path.txt
   
plots:
	ompl_benchmark_statistics.py Point0.log -d Point0.db -p Point0.pdf
	ompl_benchmark_statistics.py Point1.log -d Point1.db -p Point1.pdf
	ompl_benchmark_statistics.py Circle0.log -d Circle0.db -p Circle0.pdf
	ompl_benchmark_statistics.py Circle1.log -d Circle1.db -p Circle1.pdf
	ompl_benchmark_statistics.py Square0.log -d Square0.db -p Square0.pdf
	ompl_benchmark_statistics.py Square1.log -d Square1.db -p Square1.pdf

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $(INCLUDE_FLAGS) $< -o $@
