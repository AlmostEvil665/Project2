# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
import sys

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        "Here we check if a ghost is within range to merk us"
        "Converted this to also keep track of the closest ghost"
        mingDist = sys.maxsize
        for i in range(0, len(newGhostStates)):
            currdist = manhattanDistance(newPos, newGhostStates[i].getPosition())
            if currdist <= 1:
                return -sys.maxsize
            mingDist = min(mingDist, currdist)

        "Find the farthest piece of food from us rn"
        "Nvm too time complexity, just find a piece of food and check dist (wanna min this dist)"
        fMax = -sys.maxsize
        for foodloc in getFood(newFood):
            fMax = max(fMax, manhattanDistance(newPos, foodloc), 0)

        if newScaredTimes[0] == 0:
            return successorGameState.getScore() - fMax + mingDist
        else:
            return successorGameState.getScore() + newScaredTimes[0] - mingDist - fMax


def getFood(foodData):
    foodLocs = []

    for i in range(0, foodData.width):
        for k in range(0, foodData.height):
            if foodData[i][k]:
                foodLocs.append((i, k))
                return foodLocs

    return foodLocs


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    """ Maximize the score value and return """

    def value(self, state, depth, agent):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state), None

        if agent == 0:
            # Pacman
            max_val = -sys.maxsize
            max_action = None
            for actions in state.getLegalActions(0):
                next_state = state.generateSuccessor(0, actions)
                curr_value = self.value(next_state, depth, agent + 1)[0]
                if max_val <= curr_value:
                    max_val = curr_value
                    max_action = actions
            return max_val, max_action
        else:
            # Scary Boi
            min_val = sys.maxsize
            min_action = None
            for actions in state.getLegalActions(agent):
                next_state = state.generateSuccessor(agent, actions)
                if agent == state.getNumAgents() - 1:
                    curr_value = self.value(next_state, depth - 1, 0)[0]
                else:
                    curr_value = self.value(next_state, depth, agent + 1)[0]
                if min_val >= curr_value:
                    min_val = curr_value
                    min_action = actions
            return min_val, min_action

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        """value = self.value(gameState, self.depth)

        max_action = gameState.getLegalActions(0)
        max_value = -sys.maxsize
        for actions in gameState.getLegalActions(0):
            curr_value = self.value(gameState.generateSuccessor(0, actions))
            if curr_value >= max_value:
                max_value = curr_value
                max_action = actions"""
        result = self.value(gameState, self.depth, 0)
        value = result[0]
        max_action = result[1]
        # max_value = -sys.maxsize
        # max_action = gameState.getLegalActions(0)[0]
        # # for all possible actions
        # for actions in gameState.getLegalActions(0):
        #     # get value of each action
        #     current_state = gameState.generateSuccessor(0, actions)
        #     if len(current_state.getLegalActions(0)) == 0:
        #         continue
        #     curr_value = self.value(current_state, self.depth, False)
        #     if curr_value >= max_value:
        #         max_value = curr_value
        #         max_action = actions
        # return action which gave highest value

        return max_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        result = self.value(gameState, self.depth, 0, -sys.maxsize, sys.maxsize)
        value = result[0]
        max_action = result[1]
        # max_value = -sys.maxsize
        # max_action = gameState.getLegalActions(0)[0]
        # # for all possible actions
        # for actions in gameState.getLegalActions(0):
        #     # get value of each action
        #     current_state = gameState.generateSuccessor(0, actions)
        #     if len(current_state.getLegalActions(0)) == 0:
        #         continue
        #     curr_value = self.value(current_state, self.depth, False)
        #     if curr_value >= max_value:
        #         max_value = curr_value
        #         max_action = actions
        # return action which gave highest value

        return max_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def value(self, state, depth, agent):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state), None

        if agent == 0:
            # Pacman
            max_val = -sys.maxsize
            max_action = None
            for actions in state.getLegalActions(0):
                next_state = state.generateSuccessor(0, actions)
                curr_value = self.value(next_state, depth, agent + 1)[0]
                if max_val <= curr_value:
                    max_val = curr_value
                    max_action = actions
            return max_val, max_action
        else:
            # Scary Boi
            avg_val = 0
            num_actions = len(state.getLegalActions(agent))
            min_action = None
            for actions in state.getLegalActions(agent):
                next_state = state.generateSuccessor(agent, actions)
                if agent == state.getNumAgents() - 1:
                    curr_value = self.value(next_state, depth - 1, 0)[0]
                else:
                    curr_value = self.value(next_state, depth, agent + 1)[0]
                avg_val = avg_val + curr_value

            return avg_val / num_actions, min_action

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        result = self.value(gameState, self.depth, 0)
        value = result[0]
        expecti_action = result[1]

        return expecti_action


class Queue:
    "A container with a first-in-first-out (FIFO) queuing policy."
    def __init__(self):
        self.list = []

    def push(self,item):
        "Enqueue the 'item' into the queue"
        self.list.insert(0,item)

    def pop(self):
        """
          Dequeue the earliest enqueued item still in the queue. This
          operation removes the item from the queue.
        """
        return self.list.pop()

    def size(self):
        """
            returns the size of the list backing the queue.
        """
        return len(self.list)

    def isEmpty(self):
        "Returns true if the queue is empty"
        return len(self.list) == 0

def breadthFirstSearch(pacLoc, destination, gameState):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    visited = []
    queue = Queue()
    paths = {
        gameState: []
    }

    queue.push(gameState)
    visited.append(gameState)

    while not queue.isEmpty():
        state = queue.pop()

        # if queue.size() == 30:
        #     return None

        if state.data.agentStates[0].configuration.pos == destination:
            return len(paths[state])

        for actions in state.getLegalActions(0):
            next_state = state.generateSuccessor(0, actions)
            if next_state not in visited:
                queue.push(next_state)
                visited.append(next_state)
                paths[next_state] = paths[state].copy()
                paths[next_state].append(actions)

    return sys.maxsize

def mDistance(pos1, pos2, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = pos1
    xy2 = pos2
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])


def closestFood(position, foodLocations):
    minDistance = sys.maxsize
    minFood = position
    for c in foodLocations:
        current_distance = mDistance(position, c)
        if current_distance < minDistance:
            minDistance = current_distance
            minFood = c

    return minFood

def closestCapsule(position, capsules):
    minDistance = sys.maxsize
    minCapsule = None
    for c in capsules:
        current_distance = mDistance(position, c)
        if current_distance < minDistance:
            minDistance = current_distance
            minCapsule = c

    return minCapsule

def closestGhost(position, gameState):
    # Find all ghost locations
    ghost_locs = []
    for index in range(1, len(gameState.data.agentStates)):
        ghost_locs.append(gameState.data.agentStates[index].configuration.pos)

    minDistance = sys.maxsize
    ghost = None
    for c in ghost_locs:
        current_distance = mDistance(position, c)
        if current_distance < minDistance:
            minDistance = current_distance
            ghost = c

    return ghost

def getSuccessors(gameState):
    successors = []
    for actions in gameState.getLegalActions(0):
        successors.append(gameState.generateSuccessor(0, actions))
    return successors

def maxScaredTimer(gameState):
    # Find all ghost locations
    scaredTimes = []
    for index in range(1, len(gameState.data.agentStates)):
        scaredTimes.append(gameState.data.agentStates[index].scaredTimer)
    
    maxTime = 0
    for times in scaredTimes:
        maxTime = max(maxTime, times)
        
    return maxTime


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacLoc = currentGameState.data.agentStates[0].configuration.pos
    food = closestFood(pacLoc, currentGameState.data.food.asList())
    capsule = closestCapsule(pacLoc, currentGameState.data.capsules)
    pacAgent = currentGameState.data.agentStates[0]
    ghost = closestGhost(pacLoc, currentGameState)
    scaredTime = maxScaredTimer(currentGameState)
    
    # if scaredTime > 0:
    #     return currentGameState.data.score - mDistance(pacLoc, ghost)
    # else:
    return currentGameState.data.score - len(currentGameState.data.capsules) * 200 - mDistance(pacLoc, food)


# Abbreviation
better = betterEvaluationFunction
