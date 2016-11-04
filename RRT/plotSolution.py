from mpl_toolkits.mplot3d import Axes3D, art3d
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
from math import sin, cos

# Plot a path in R2
def plotR2(path, environment, robot):
    fig = plt.figure()
    ax = fig.gca()
    x0, y0, dx, dy = ax.get_position().bounds
    maxd = max(dx, dy)
    width = 6 * maxd / dx
    height = 6 * maxd / dy
    fig.set_size_inches((width, height))

    if environment == '0': # L shaped obstacle
        ax.add_patch(patches.Polygon([(-0.5,0.5),(-0.5,-0.5),(-0.25,-0.5),(-0.25,0.5)], fill=True, color='0.20'))
        ax.add_patch(patches.Polygon([(0.25,-0.25),(0.25,-0.5),(-0.25,-0.5),(-0.25,-0.25)], fill=True, color='0.20'))
    elif environment == '1': # Two rectangles with space in between
        ax.add_patch(patches.Polygon([(-1.0,0.5),(-1.0,-0.5),(-0.25,-0.5),(-0.25,0.5)], fill=True, color='0.20'))
        ax.add_patch(patches.Polygon([(1.0,0.5),(1.0,-0.5),(0.25,-0.5),(0.25,0.5)], fill=True, color='0.20'))

    # Plotting the path
    X = [p[0] for p in path]
    Y = [p[1] for p in path]
    ax.plot(X, Y)

    if robot == '1': # circle robot
        for p in path:
            ax.add_patch(patches.Circle((p[0],p[1]), 0.1, fill = False, color = '0.20'))

    plt.axis([-1,1,-1,1])
    plt.show()

def plotSE2(path, environment, robot):
    fig = plt.figure()
    ax = fig.gca()
    x0, y0, dx, dy = ax.get_position().bounds
    maxd = max(dx, dy)
    width = 6 * maxd / dx
    height = 6 * maxd / dy
    fig.set_size_inches((width, height))

    # Drawing the unit square
    if environment == '0': # L shaped obstacle
        ax.add_patch(patches.Polygon([(-0.5,0.5),(-0.5,-0.5),(-0.25,-0.5),(-0.25,0.5)], fill=True, color='0.20'))
        ax.add_patch(patches.Polygon([(0.25,-0.25),(0.25,-0.5),(-0.25,-0.5),(-0.25,-0.25)], fill=True, color='0.20'))
    elif environment == '1': # Two rectangles with space in between
        ax.add_patch(patches.Polygon([(-1.0,0.5),(-1.0,-0.5),(-0.25,-0.5),(-0.25,0.5)], fill=True, color='0.20'))
        ax.add_patch(patches.Polygon([(1.0,0.5),(1.0,-0.5),(0.25,-0.5),(0.25,0.5)], fill=True, color='0.20'))

    # Plotting the path (reference point)
    X = [p[0] for p in path]
    Y = [p[1] for p in path]
    ax.plot(X, Y)

    # Plotting the square as hollow shapes with side length 2*hside
    hside = 0.125
    lineSegs = [[[-hside, hside], [hside, hside]], [[-hside, -hside], [hside, -hside]], [[-hside, hside], [-hside, -hside]], [[hside, hside], [hside, -hside]]]
    for lineSeg in lineSegs:
        linePath = []
        for p in path:
            x = []
            y = []
            for v in lineSeg:
                x.append(v[0] * cos(p[2]) - v[1] * sin(p[2]) + p[0])
                y.append(v[0] * sin(p[2]) + v[1] * cos(p[2]) + p[1])

            ax.plot(x, y, 'k')


    plt.axis([-1,1,-1,1])
    plt.show()


# Read the cspace definition and the path from filename
def readPath(filename):
    lines = [line.rstrip() for line in open(filename) if len(line.rstrip()) > 0]

    if len(lines) == 0:
        print "Empty File"
        sys.exit(1)

    cspace = lines[0].strip()
    environment = lines[1].strip()
    robot = lines[2].strip()
    if (cspace != 'R2' and cspace != 'R3' and cspace != 'SE2'):
        print "Unknown c-space identifier: " + cspace
        sys.exit(1)

    data = [[float(x) for x in line.split(' ')] for line in lines[3:]]
    return cspace, data, environment, robot

if __name__ == '__main__':
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = 'path.txt'

    cspace, path, environment, robot = readPath(filename)

    if cspace == 'R2':
        plotR2(path, environment, robot)
    elif cspace == 'R3':
        plotR3(path)
    elif cspace == 'SE2':
        plotSE2(path, environment, robot)
