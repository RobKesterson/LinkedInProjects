"""
Robert Kesterson

Pathfinding A* search based on the Tanimoto algorithm for minimal path.

This implementation uses the shapely software package (installation required) to represent a robot navigating
a room filled with rectangular objects. The code first loads a map of the room containing the start and end
points as well as a count of the number of obstacles to be navigated. Each rectangular obstacle is represented
as a series of four points. Please see provided text files for formatting.

Included are three examples with rooms of varying complexity and size.

The Taninmoto pathfinding algorithm is a classic for navigation software. It works by utilizing an open / close
list pair, representing the possible movements as instances of states which are moved between the two lists.

The open list contains states representing paths being built and the close list represents the next possible
movement for each path. The shortest path is guaranteed by comapring gValues (the cost of the path from the
initial state to the current one).
"""

import heapq
import shapely
import shapely.geometry
import math


# state class to keep track of all required information
class state:
    pointCoordinates = "0,0"
    # cost of the path from the initial state to this state
    gValue = 0.0
    # estimates cost from this state to the goal
    hValue = 0.0
    # sum of gValue and hValue
    fValue = 0.0
    parentState = "0,0"
    successorList = []

    def __init__(self, pointCoord, gVal, hVal, parentSt):
        self.pointCoordinates = pointCoord
        self.gValue = gVal
        self.hValue = hVal
        self.fValue = gVal + hVal
        self.parentState = parentSt

    def getPointCoordinates(self):
        return self.pointCoordinates


def loadMapFile(path):
    desiredMap = open(path, 'r')
    returnArray = []
    # startState = returnArray[0]
    returnArray.append(desiredMap.readline().strip('\n').replace(' ', ','))
    # goalState = returnArray[1]
    returnArray.append(desiredMap.readline().strip('\n').replace(' ', ','))
    obstacleCount = desiredMap.readline().strip('\n')
    obstaclePolygonArray = []
    obstacleStringArray = []
    # read in coordinates of obstacles
    currentLine = desiredMap.readline().strip('\n').replace(' ', ',')
    while currentLine:
        x = currentLine.split(',')
        obstacleStringArray.append(x[0] + "," + x[1])
        obstacleStringArray.append(x[2] + "," + x[3])
        obstacleStringArray.append(x[4] + "," + x[5])
        obstacleStringArray.append(x[6] + "," + x[7])
        shape1 = shapely.geometry.Polygon(
            [[int(x[0]), int(x[1])], [int(x[2]), int(x[3])],
             [int(x[4]), int(x[5])], [int(x[6]), int(x[7])]])
        obstaclePolygonArray.append(shape1)
        currentLine = desiredMap.readline().strip('\n').replace(' ', ',')

    desiredMap.close()
    returnArray.append(obstaclePolygonArray)
    returnArray.append(obstacleStringArray)
    return returnArray

# not strict requirements for recursive input. Just output path must be correct. No requirement for recursion
def recursiveASearch(searchHeap):
    global closeList
    global priorityQueue
    if searchHeap.__len__() == 0:
        return "no solution"
    else:
        # 3. find and remove the item [s,f] on OPEN having lowest f (sum of cost of the path + projected distance).
        workingState = heapq.heappop(searchHeap)

        # Put [fValue, state] on CLOSED
        closeList.append(workingState)
        # if s is a goal state: output its description (and backtrace a path) and if h is known to be admissible, halt
        if workingState[1].pointCoordinates.__eq__(goalState):
            # generate solution based on backtrace with parentNode pointers
            solutionPath = []
            rootFound = False
            solutionPath.append((workingState[1].pointCoordinates, workingState[1].gValue))
            nodeBacktracer = workingState[1].parentState
            while not rootFound:
                if nodeBacktracer.pointCoordinates.__eq__(startState):
                    rootFound = True
                    solutionPath.append((startState, 0))
                else:
                    solutionPath.append((nodeBacktracer.pointCoordinates, nodeBacktracer.gValue))
                    nodeBacktracer = nodeBacktracer.parentState
            # reverse solutionPath to make the results human-readable and support desired output format
            solutionPath.reverse()
            return solutionPath
        # else - generate list L of successors with precomputed f values f(s') = g(s') + h(s')
        else:
            # list L is returnList
            returnList = []

            # build ordered list of points for Shapely LineString object of current path
            pointList = []

            if (workingState[1].pointCoordinates.__eq__(startState)):
                pointList.append([int(startPoints[0]), int(startPoints[1])])
            else:
                rootFound = False
                x = workingState[1].pointCoordinates.split(',')
                pointList.append([int(x[0]), int(x[1])])
                nodeBacktracer = workingState[1].parentState
                while not rootFound:
                    if nodeBacktracer.pointCoordinates.__eq__(startState):
                        rootFound = True
                        pointList.append([int(startPoints[0]), int(startPoints[1])])
                    else:
                        a = nodeBacktracer.pointCoordinates.split(',')
                        pointList.append([int(a[0]), int(a[1])])
                        nodeBacktracer = nodeBacktracer.parentState

            pointList.reverse()

            # collision detection step
            for objectSelect in objectStrings:
                # if objectSelect != last item in returnList, then we know we can begin constructing the LineString
                if not objectSelect.__eq__(pointList[pointList.__len__() - 1].__str__().strip('[').strip(']').replace(', ', ',')):
                    y = objectSelect.split(',')
                    pointList.append([int(y[0]), int(y[1])])
                    # create LineString object
                    testLine = shapely.geometry.LineString(pointList)
                    collisionDetected = False
                    collisionObjectCount = 0
                    # Sadly the testLine.intersects(object) function does not work as expected
                    # ,so we must use a combination of .contains and .crosses functions
                    while collisionObjectCount < objectPolygons.__len__() and not collisionDetected:
                        # First we check to see if the desired polygon contains the test line
                        collisionDetected = objectPolygons[collisionObjectCount].contains(testLine)
                        # Second we check to see if the testline crosses the desired polygon
                        if not collisionDetected:
                            collisionDetected = testLine.crosses(objectPolygons[collisionObjectCount])
                        collisionObjectCount += 1
                    # If there is no collision, then we know we have a valid path and can create the next state
                    if not collisionDetected:
                        # compute state and f value
                        gValue = testLine.length
                        hValue = math.sqrt((int(goalPoints[0]) - int(y[0])) ** 2 + (int(goalPoints[1]) - int(y[1])) ** 2)
                        nextState = state(objectSelect, gValue, hValue, workingState[1])
                        fValue = gValue + hValue
                        returnList.append((fValue, nextState))

                    # remove the last item from pointList to check next possible point
                    pointList.pop()

            # for x in L (returnList) perform Tanimoto's comparisons with the OPEN and CLOSE
            listLoopCounter = 0
            while listLoopCounter < returnList.__len__():
                foundOnClose = False
                # if there is a pair on closeList
                for closeStates in closeList:
                    if returnList[listLoopCounter][1].pointCoordinates.__eq__(closeStates[1].pointCoordinates):
                        foundOnClose = True
                        if returnList[listLoopCounter][0] > closeStates[0]:
                            returnList.remove(returnList[listLoopCounter])
                            listLoopCounter = 0
                        elif returnList[listLoopCounter][0] <= closeStates[0]:
                            closeList.remove(closeStates)
                            listLoopCounter += 1
                # else if there is already a pair on openList
                if not foundOnClose:
                    searchHeapCounter = 0
                    foundOnOpen = False
                    while searchHeapCounter < searchHeap.__len__() and not foundOnOpen:
                        if returnList[listLoopCounter][1].pointCoordinates.__eq__(searchHeap[searchHeapCounter][1].pointCoordinates):
                            if returnList[listLoopCounter][0] > searchHeap[searchHeapCounter][0]:
                                returnList.remove(returnList[listLoopCounter])
                                #listLoopCounter = 0
                                foundOnOpen = True
                            elif returnList[listLoopCounter][0] <= searchHeap[searchHeapCounter][0]:
                                searchHeap.remove(searchHeap[searchHeapCounter])
                                heapq.heapify(searchHeap)
                            searchHeapCounter = 0
                        else:
                            searchHeapCounter += 1
                    listLoopCounter += 1
                    if foundOnOpen:
                        listLoopCounter = 0
                #listLoopCounter += 1

            """
            edge case for last step - if we are at (25,25), then (25,25) -> (25,33) -> (25,35) and (25,25) -> (25,35)
            have the same fValue which will cause a collision while performing the push operation on the heap
            to prevent this, we add a check for (25,25) to remove the (25,25) -> (25,33) -> (25,35) path
            from the returnList
            """
            if workingState[1].pointCoordinates.__eq__("25,25"):
                returnList.remove(returnList[2])

            # insert all remaining members of L onto openList
            for validStates in returnList:
                # push the tuple onto the heap
                # f value must go first in tuple pair when pushing onto the heap because the heap will sort based on
                # first value
                heapq.heappush(searchHeap, validStates)
    return recursiveASearch(searchHeap)


# required close list
closeList = []

mapList = ['PathfindingASearch\map1.txt', 'PathfindingASearch\map2.txt', 'PathfindingASearch\map3.txt']

for mapSelect in mapList:
    mapGuts = loadMapFile(mapSelect)
    startState = mapGuts[0]
    startPoints = startState.split(',')
    goalState = mapGuts[1]
    goalPoints = goalState.split(',')
    objectPolygons = mapGuts[2]
    objectStrings = mapGuts[3]

    hEstimate = math.sqrt((int(goalPoints[0]) - int(startPoints[0])) ** 2 + (int(goalPoints[1]) - int(startPoints[1])) ** 2)

    firstState = state(startState, 0.0, hEstimate, startState)

    # required open list using a priority queue implemented as a heap in Python
    priorityQueue = [(hEstimate, firstState)]
    heapq.heapify(priorityQueue)
    results = recursiveASearch(priorityQueue)
    if results.__len__() > 0:
        print("Solution identified for: " + mapSelect)
        print("Point   Cumulative Cost \b")

        for solutionNodes in results:
            print(solutionNodes[0] + "   (" + solutionNodes[1].__round__(3).__str__() + ") \b")
        print("\n")
    else:
        print("No solution identified for this data set")