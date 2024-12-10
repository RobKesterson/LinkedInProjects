"""
Robert Kesterson

Code produces solutions to 3 monks / 3 cannibals problem - one of the all-time classic examples for introductory
ai. I chose to use all native data structures and a recursive search. The state stack is represented by a Python
list. The states are represented as a string which is the sole input in the recursive function call.

State notation is (monks on left bank, cannibals on left bank, which side of the lake the boat is on).

This version of monks / cannibals problem assumes the following:
3 monks and 3 cannibals are on the left side of a lake and want to cross using a boat with capacity for
2 passengers
If either side of the lake has more cannibals than monks, the cannibals eat the monks
The boat must always have at least one passenger to move
The monks would like to remain alive

The solutions are represented as an ordered list of states.
"""


# counters for search
# counts the states where the cannibals eat the missionaries
illegal_state_counter = 0
# counts the states that are repeated
repeated_state_counter = 0
# counts the total number of searched states
total_search_counter = 0

# ordered list for correct solutions
solutionList = []
# stack to keep track of the path above the current state
searchStack = []


# search is a recursive procedure that is called with the start node and has arg s (state)
# initially the state is the start state
def recursive_missionary_cannibal_search(state):
    global illegal_state_counter
    global repeated_state_counter
    global total_search_counter
    global searchStack
    global solutionList
    mCounterRight = 3 - int(state[0])

    # checks to determine if the current state is the same as an ancestor state on the same path
    if not searchStack.__contains__(state):
        searchStack.append(state)
        # checks first if the state we have arrived at (s) is the goal
        if int(state[0]) == 0 and int(state[2]) == 0:
            total_search_counter += 1
            solutionString = ""
            for x in searchStack:
                solutionString += "(" + x + ")"
            solutionList.append(solutionString)
            searchStack.pop()
        # checks to determine if the cannibals on the right bank eat the missionaries
        elif 3 - int(state[0]) < 3 - int(state[2]) and mCounterRight != 0:
            illegal_state_counter += 1
            searchStack.pop()
        # checks to determine if the cannibals on the left bank eat the missionaries
        elif int(state[0]) < int(state[2]) and int(state[0]) != 0:
            illegal_state_counter += 1
            searchStack.pop()
        else:
            total_search_counter += 1
            successorList = []
            m_count = int(state[0])
            c_count = int(state[2])
            # generates a list of successors to the current state
            if state[4] == "L":
                # MM
                if m_count - 2 >= 0:
                    successorList.append(str(m_count - 2) + "," + str(c_count) + ",R")
                # CC
                if c_count - 2 >= 0:
                    successorList.append(str(m_count) + "," + str(c_count - 2) + ",R")
                # MC
                if m_count - 1 >= 0 and c_count - 1 >= 0:
                    successorList.append(str(m_count - 1) + "," + str(c_count - 1) + ",R")
                # M
                if m_count - 1 >= 0:
                    successorList.append(str(m_count - 1) + "," + str(c_count) + ",R")
                # C
                if c_count - 1 >= 0:
                    successorList.append(str(m_count) + "," + str(c_count - 1) + ",R")
            if state[4] == "R":
                # MM
                if 0 <= m_count <= 1:
                    successorList.append(str(m_count + 2) + "," + str(c_count) + ",L")
                # CC
                if 0 <= c_count <= 1:
                    successorList.append(str(m_count) + "," + str(c_count + 2) + ",L")
                # MC
                if m_count - 1 >= 0 and c_count - 1 >= 0 and m_count != 3 and c_count != 3:
                    successorList.append(str(m_count + 1) + "," + str(c_count + 1) + ",L")
                # M
                if 2 >= m_count >= 0:
                    successorList.append(str(m_count + 1) + "," + str(c_count) + ",L")
                # C
                if 2 >= c_count >= 0:
                    successorList.append(str(m_count) + "," + str(c_count + 1) + ",L")
            # the search iterates through list L, calling itself recursively for each state in L
            for validStates in successorList:
                recursive_missionary_cannibal_search(validStates)
            searchStack.pop()
    else:
        repeated_state_counter += 1

def showResults():
    print("The search found " + str(len(solutionList)) + " solutions")
    print("The solutions are:\b")
    for y in solutionList:
        print(y + " \b")
    print("The illegal state count was " + str(illegal_state_counter) + ".")
    print("The repeated state count was " + str(repeated_state_counter) + ".")
    print("The total count of valid states searched was " + str(total_search_counter))

recursive_missionary_cannibal_search("3,3,L")
showResults()