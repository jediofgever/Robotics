#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <math.h>

#include <boost/bind.hpp>

// Including SimpleSetup.h will pull in MOST of what you need to plan
#include <ompl/geometric/SimpleSetup.h>
// Except for the state space definitions and any planners
#include <ompl/geometric/planners/prm/PRM.h>
#include <ompl/geometric/planners/est/EST.h>
#include <ompl/geometric/planners/rrt/RRT.h>

#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/spaces/SE2StateSpace.h>
#include <ompl/base/spaces/SO2StateSpace.h>
#include <ompl/tools/benchmark/Benchmark.h>
#include <omplapp/config.h>

#include "randomtree.h"

typedef std::pair<double, double> Point2D;
typedef std::vector<Point2D> Rect;

const int CHOICES = 3;
const int ENVIRONMENTS = 2;
const double epsilon = 0.01;
const double radius = 0.1;
const double square = 0.125;

double dist(double x1, double y1, double x2, double y2)
{
    return sqrt(pow(x1-x2, 2) + pow(y1-y2, 2));
}

bool lineIntersection(Point2D ours0, Point2D ours1, Point2D theirs0, Point2D theirs1)
{
    double theirs_lowerX = std::min(theirs0.first, theirs1.first);
    double theirs_upperX = std::max(theirs0.first, theirs1.first);
    double ours_lowerX = std::min(ours0.first, ours1.first);
    double ours_upperX = std::max(ours0.first, ours1.first);

    // Check if Y range of the lines overlap
    double theirs_lowerY = std::min(theirs0.second, theirs1.second);
    double theirs_upperY = std::max(theirs0.second, theirs1.second);
    double ours_lowerY = std::min(ours0.second, ours1.second);
    double ours_upperY = std::max(ours0.second, ours1.second);

    bool y0_overlap = (ours_lowerY >= theirs_lowerY) && (ours_lowerY <= theirs_upperY);
    bool y1_overlap = (ours_upperY >= theirs_lowerY) && (ours_lowerY <= theirs_upperY);
    if(!(y0_overlap || y1_overlap))
        return false;

    double ours_m = (ours1.second - ours0.second) / (ours1.first - ours0.first);
    double ours_b = ours0.second - ours_m * ours0.first;

    double theirs_m = (theirs1.second - theirs0.second) / (theirs1.first - theirs0.first);
    double theirs_b = theirs0.second - theirs_m * theirs0.first;

    if(isinf(ours_m))
    {
        // Check if X range of the lines overlap
        bool x_overlap = (theirs_lowerX < ours0.first) && (theirs_upperX > ours0.first);
        if(!x_overlap)
            return false;

        double theirs_value = theirs_m * ours0.first + theirs_b;
        return (theirs_value >= ours_lowerY) && (theirs_value <= ours_upperY);
    }

    if(isinf(theirs_m))
    {
        // Check if X range of the lines overlap
        bool x_overlap = (ours_lowerX < theirs0.first) && (ours_upperX > theirs0.first);
        if(!x_overlap)
            return false; 

        double ours_value = ours_m * theirs0.first + ours_b;
        return (ours_value >= theirs_lowerY) && (ours_value <= theirs_upperY);
    }

    // Brute-Force
    for(double pos = ours0.first; pos < ours1.first; pos+=epsilon)
    {
        if(pos >= theirs_lowerX && pos <= theirs_upperX)
        {
            double ours_value = ours_m * pos + ours_b;
            double theirs_value = theirs_m * pos + theirs_b;
            double diff = ours_value - theirs_value;
            if(abs(diff) < epsilon)
                return true;
        }
    }
    return false;
}

// Default state validity checker.  It says every state is valid.
bool stateAlwaysValid(const ompl::base::State* /*state*/)
{
    return true;
}

bool stateValidPointRobot(const ompl::base::State* state, const double minBound, const double maxBound, const std::vector<Rect> obstacles)
{
    const ompl::base::RealVectorStateSpace::StateType* r2state;
    r2state = state->as<ompl::base::RealVectorStateSpace::StateType>();

    double x = r2state->values[0];
    double y = r2state->values[1];

    if(x < minBound || y < minBound || x > maxBound || y > maxBound)
        return false;

    for(Rect r : obstacles)
    {
        if(x >= r[0].first && x <= r[2].first && y >= r[0].second && y <= r[2].second)
            return false;
    }

    return true;
}

bool stateValidCircleRobot(const ompl::base::State* state, const double minBound, const double maxBound, const std::vector<Rect> obstacles)
{
    const ompl::base::RealVectorStateSpace::StateType* r2state;
    r2state = state->as<ompl::base::RealVectorStateSpace::StateType>();

    double x = r2state->values[0];
    double y = r2state->values[1];

    if(x < minBound || y < minBound || x > maxBound || y > maxBound)
        return false;

    for(const Rect& r : obstacles)
    {
        if(x >= r[0].first-radius && x <= r[2].first+radius && y >= r[0].second && y <= r[2].second)
        {
            return false;
        }
        else if(x >= r[0].first && x <= r[2].first && y >= r[0].second-radius && y <= r[2].second+radius)
        {
            return false;
        }
        else 
        {
            for(int i = 0; i < r.size(); ++i)
            {
                if(dist(x, y, r[i].first, r[i].second) <= radius)
                    return false;
            }
        }
    }

    return true;
}

bool stateValidSquareRobot(const ompl::base::State* state, const double minBound, const double maxBound, const std::vector<Rect> obstacles)
{
    const ompl::base::CompoundState* cstate;
    cstate = state->as<ompl::base::CompoundState>();
    const ompl::base::RealVectorStateSpace::StateType* r2state;
    r2state = cstate->as<ompl::base::RealVectorStateSpace::StateType>(0);
    const ompl::base::SO2StateSpace::StateType* so2state;
    so2state = cstate->as<ompl::base::SO2StateSpace::StateType>(1);

    double x = r2state->values[0];
    double y = r2state->values[1];
    double theta = so2state->value;    

    // Initial Square Robot Points
    std::vector<Point2D> pts;
    pts.push_back(std::make_pair(-square, -square));
    pts.push_back(std::make_pair(square, -square));
    pts.push_back(std::make_pair(square, square));
    pts.push_back(std::make_pair(-square, square));

    // Transform Square Robot Points to state position
    for(int i = 0; i < pts.size(); ++i)
    {
        double newX = pts[i].first * cos(theta) - pts[i].second * sin(theta) + x;
        double newY = pts[i].first * sin(theta) + pts[i].second * cos(theta) + y;

        if(newX < minBound || newY < minBound || newX > maxBound || newY > maxBound)
        {
            return false;
        }

        pts[i] = std::make_pair(newX, newY);
    }

    for(const Rect& r : obstacles)
    {
        // None of the points of square robot are contained inside of the obstacle 
        /*
           for(int j = 0; j < pts.size(); ++j)
           {
           if(pts[j].first >= r[0].first && pts[j].first <= r[2].first && pts[j].second >= r[0].second && pts[j].second <= r[2].second)
           return false;
           }
         */

        // Edge of rectangle r
        for(int i = 0; i < r.size(); ++i)
        {
            // Edge of square robot
            for(int j = 0; j < pts.size(); ++j)
            {
                bool intersection = lineIntersection(pts[j], pts[(j+1) % pts.size()], r[i], r[(i+1) % r.size()]);
                if(intersection)
                {
                    return false;
                }
            }
        }
    }
    return true;
}


void planWithSimpleSetupR2(int environment, int robot, std::vector<Rect> obstacles, std::string title = "Default", bool benchmark = false)
{
    // Step 1) Create the state (configuration) space for your system
    // For a robot that can translate in the plane, we can use R^2 directly
    // We also need to set bounds on R^2
    ompl::base::StateSpacePtr r2(new ompl::base::RealVectorStateSpace(2));

    // We need to set bounds on R^2
    ompl::base::RealVectorBounds bounds(2);
    bounds.setLow(-1); // x and y have a minimum of -2
    bounds.setHigh(1); // x and y have a maximum of 2

    // Cast the r2 pointer to the derived type, then set the bounds
    r2->as<ompl::base::RealVectorStateSpace>()->setBounds(bounds);

    // Step 2) Create the SimpleSetup container for the motion planning problem.
    // this container ensures that everything is initialized properly before
    // trying to solve the motion planning problem.
    // ALWAYS USE SIMPLE SETUP!  There is no loss of functionality when using
    // this class.
    ompl::geometric::SimpleSetup ss(r2);

    // Step 3) Setup the StateValidityChecker
    // This is a function that takes a state and returns whether the state is a
    // valid configuration of the system or not.  For geometric planning, this
    // is a collision checker
    switch(robot)
    {
        case 0:
            ss.setStateValidityChecker(boost::bind(stateValidPointRobot, _1, -1, 1, obstacles));
            break;
        case 1:
            ss.setStateValidityChecker(boost::bind(stateValidCircleRobot, _1, -1, 1, obstacles));
            break;
        default:
            ss.setStateValidityChecker(stateAlwaysValid);
            break;
    }

    // Step 4) Specify the start and goal states
    // ScopedState creates the correct state instance for the state space
    ompl::base::ScopedState<> start(r2);
    start[0] = -0.9;
    start[1] = -0.9;

    ompl::base::ScopedState<> goal(r2);
    goal[0] = 0.9;
    goal[1] = 0.9;

    // set the start and goal states
    ss.setStartAndGoalStates(start, goal);

    if(benchmark)
    {
        ompl::tools::Benchmark b(ss, title);
        b.addPlanner(ompl::base::PlannerPtr(new ompl::geometric::randomtree(ss.getSpaceInformation())));
        b.addPlanner(ompl::base::PlannerPtr(new ompl::geometric::PRM(ss.getSpaceInformation())));
        b.addPlanner(ompl::base::PlannerPtr(new ompl::geometric::EST(ss.getSpaceInformation())));
        b.addPlanner(ompl::base::PlannerPtr(new ompl::geometric::RRT(ss.getSpaceInformation())));

        ompl::tools::Benchmark::Request req;
        req.maxTime = 30.0;
        req.maxMem = 1000.0;
        req.runCount = 20;
        req.displayProgress = true;
        b.benchmark(req);
        std::string logfile = title + ".log";
        b.saveResultsToFile(logfile.c_str());
    }
    else
    {
        // Step 5) (optional) Specify a planning algorithm to use
        ompl::base::PlannerPtr planner(new ompl::geometric::randomtree(ss.getSpaceInformation()));
        ss.setPlanner(planner);

        // Step 6) Attempt to solve the problem within the given time (seconds)
        ompl::base::PlannerStatus solved = ss.solve(1.0);

        if (solved)
        {
            // Apply some heuristics to simplify (and prettify) the solution
            ss.simplifySolution();

            // print the path to screen
            std::cout << "Found solution:" << std::endl;
            ompl::geometric::PathGeometric& path = ss.getSolutionPath();
            path.interpolate(50);
            path.printAsMatrix(std::cout);

            // print path to file
            std::ofstream fout("path.txt");
            fout << "R2" << std::endl;
            fout << environment << std::endl;
            fout << robot << std::endl;
            path.printAsMatrix(fout);
            fout.close();
        }
        else
            std::cout << "No solution found" << std::endl;
    }
}

void planWithSimpleSetupSE2(int environment, int robot, std::vector<Rect> obstacles, std::string title = "Default", bool benchmark = false)
{
    // Step 1) Create the state (configuration) space for your system
    // In this instance, we will plan for a unit-length line segment
    // that both translates and rotates in the plane.
    // The state space can be easily composed of simpler state spaces
    ompl::base::StateSpacePtr se2(new ompl::base::SE2StateSpace());
    ompl::base::RealVectorBounds bounds(2);
    bounds.setLow(-1);
    bounds.setHigh(1);
    se2->as<ompl::base::SE2StateSpace>()->setBounds(bounds);

    // Step 2) Create the SimpleSetup container for the motion planning problem.
    // this container ensures that everything is initialized properly before
    // trying to solve the motion planning problem using OMPL.
    // ALWAYS USE SIMPLE SETUP!  There is no loss of functionality when using
    // this class.
    ompl::geometric::SimpleSetup ss(se2);

    // Step 3) Setup the StateValidityChecker
    // This is a function that takes a state and returns whether the state is a
    // valid configuration of the system or not.  For geometric planning, this
    // is a collision checker
    ss.setStateValidityChecker(boost::bind(stateValidSquareRobot, _1, -1, 1, obstacles));

    // Step 4) Specify the start and goal states
    // ScopedState creates the correct state instance for the state space
    // You can index into the components of the state easily with ScopedState
    // The indexes correspond to the order that the StateSpace components were
    // added into the StateSpace
    ompl::base::ScopedState<> start(se2);
    start[0] = -0.75;
    start[1] = -0.75;
    start[2] = -0.0;
    ompl::base::ScopedState<> goal(se2);
    goal[0] = 0.75;
    goal[1] = 0.75;
    goal[2] = 0.0;

    // set the start and goal states
    ss.setStartAndGoalStates(start, goal);

    if(benchmark)
    {
        // Benchmark Code - Project 3
        ompl::tools::Benchmark b(ss, title);
        b.addPlanner(ompl::base::PlannerPtr(new ompl::geometric::randomtree(ss.getSpaceInformation())));
        b.addPlanner(ompl::base::PlannerPtr(new ompl::geometric::PRM(ss.getSpaceInformation())));
        b.addPlanner(ompl::base::PlannerPtr(new ompl::geometric::EST(ss.getSpaceInformation())));
        b.addPlanner(ompl::base::PlannerPtr(new ompl::geometric::RRT(ss.getSpaceInformation())));

        ompl::tools::Benchmark::Request req;
        req.maxTime = 30.0;
        req.maxMem = 1000.0;
        req.runCount = 20;
        req.displayProgress = true;
        b.benchmark(req);
        std::string logfile = title + ".log";
        b.saveResultsToFile(logfile.c_str());
    }
    else
    {
        // Step 5) (optional) Specify a planning algorithm to use
        ompl::base::PlannerPtr planner(new ompl::geometric::randomtree(ss.getSpaceInformation()));
        ss.setPlanner(planner);

        // Step 6) Attempt to solve the problem within the given time (seconds)
        ompl::base::PlannerStatus solved = ss.solve(1.0);

        if (solved)
        {
            // Apply some heuristics to simplify (and prettify) the solution
            ss.simplifySolution();

            // print the path to screen
            std::cout << "Found solution:" << std::endl;
            ompl::geometric::PathGeometric& path = ss.getSolutionPath();
            path.interpolate(50);
            path.printAsMatrix(std::cout);

            // print path to file
            std::ofstream fout("path.txt");
            fout << "SE2" << std::endl;
            fout << environment << std::endl;
            fout << robot << std::endl;
            path.printAsMatrix(fout);
            fout.close();
        }
        else
            std::cout << "No solution found" << std::endl;
    }
}

int main(int, char **)
{
    std::vector<std::vector<Rect>> obstacles;
    std::vector<Rect> go_around_obstacles;
    std::vector<Rect> narrow_obstacles;
    std::vector<Point2D> rect1;
    std::vector<Point2D> rect2;
    std::vector<Point2D> rect3;
    std::vector<Point2D> rect4;
    // Go-Around L-Shaped Obstacle
    rect1.push_back(std::make_pair(-0.5, -0.5));
    rect1.push_back(std::make_pair(-0.25, -0.5));
    rect1.push_back(std::make_pair(-0.25, 0.5));
    rect1.push_back(std::make_pair(-0.5, 0.5));

    rect2.push_back(std::make_pair(-0.25, -0.5));
    rect2.push_back(std::make_pair(0.25, -0.5));
    rect2.push_back(std::make_pair(0.25, -0.25));
    rect2.push_back(std::make_pair(-0.25, -0.25));

    // Narrow Passage in-between two obstacles
    rect3.push_back(std::make_pair(-1.0, -0.5));
    rect3.push_back(std::make_pair(-0.25, -0.5));
    rect3.push_back(std::make_pair(-0.25, 0.5));
    rect3.push_back(std::make_pair(-1.0, 0.5));

    rect4.push_back(std::make_pair(0.25, -0.5));
    rect4.push_back(std::make_pair(1.0, -0.5));
    rect4.push_back(std::make_pair(1.0, 0.5));
    rect4.push_back(std::make_pair(0.25, 0.5));

    go_around_obstacles.push_back(rect1);
    go_around_obstacles.push_back(rect2);
    narrow_obstacles.push_back(rect3);
    narrow_obstacles.push_back(rect4);
    obstacles.push_back(go_around_obstacles);
    obstacles.push_back(narrow_obstacles);

    int choice;
    do
    {
        std::cout << "Plan for: "<< std::endl;
        std::cout << " (0) A point in 2D" << std::endl;
        std::cout << " (1) A circle in 2D" << std::endl;
        std::cout << " (2) A square in 2D" << std::endl;
        std::cout << " (3) Benchmark RandomTree Planner" << std::endl;

        std::cin >> choice;
    } while (choice < 0 || choice > CHOICES);

    if(choice == CHOICES)
    {
        planWithSimpleSetupR2(0, 0, obstacles[0], "Point0", true);
        planWithSimpleSetupR2(0, 1, obstacles[0], "Circle0", true);
        planWithSimpleSetupSE2(0, 2, obstacles[0], "Square0", true);
        planWithSimpleSetupR2(1, 0, obstacles[1], "Point1", true);
        planWithSimpleSetupR2(1, 1, obstacles[1], "Circle1", true);
        planWithSimpleSetupSE2(1, 2, obstacles[1], "Square1", true);
        return 0;
    }

    int environment;
    do
    {
        std::cout << "Pick Environment: "<< std::endl;
        std::cout << " (0) Go Around" << std::endl;
        std::cout << " (1) Narrow Path" << std::endl;

        std::cin >> environment;
    } while (environment < 0 || environment > ENVIRONMENTS);

    switch(choice)
    {
        case 0:
            planWithSimpleSetupR2(environment, choice, obstacles[environment]);
            break;
        case 1:
            planWithSimpleSetupR2(environment, choice, obstacles[environment]);
            break;
        case 2:
            planWithSimpleSetupSE2(environment, choice, obstacles[environment]);
            break;
    }
    return 0;
}
