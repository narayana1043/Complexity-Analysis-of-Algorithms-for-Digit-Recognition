from imageDataRead import np
from imageDataRead import imageDataViewer as idv
import pickle


trainDataDict = pickle.load(open('./pickles/trainDataDict.p', 'rb'))

class Node:

    def __init__(self, data):

        self.data = data

    def setNodeId(self, nodeId):

        self.nodeId = nodeId

    def setParent(self, parent=None):

        self.parent = parent

    def setChildren(self, children=None):

        self.children = children


def matching(nodeList):

    parentList = list()

    while len(nodeList) > 1:

        minScore = 28 * 255 * 255
        minIndex = 0
        pickedImg = nodeList[0]
        del nodeList[0]

        for index in range(len(nodeList)):

            x= np.subtract(pickedImg.data, nodeList[index].data, dtype=np.ndarray)
            y = np.absolute(x)
            score = np.sum(y)
            #print(score)

            if score < minScore:

                minScore = score
                minIndex = index

        #print(minIndex)
        closetImg = nodeList[minIndex]
        del nodeList[minIndex]

        avgImg = np.around(np.add(closetImg.data, pickedImg.data)/2, decimals=-1).astype(np.int16)

        pickedImg.setNodeId(nodeId='L')
        closetImg.setNodeId(nodeId='R')

        parent = Node(data=avgImg)

        pickedImg.setParent(parent=parent)
        closetImg.setParent(parent=parent)
        parent.setChildren(children= [pickedImg, closetImg])

        parentList.append(parent)

    else:

        if len(nodeList) == 1:

            parent = Node(data=nodeList[0].data)
            parent.setChildren(children=nodeList[0])
            parentList.append(parent)

    return parentList


def bottomUpBuilder(particularNumTrainData):

    pickedNum = particularNumTrainData
    nodeList = list()
    level = {}
    levelId = 0
    level[levelId] = list()

    for img in pickedNum:

        node = Node(data=img)
        nodeList.append(node)

    print(len(nodeList))
    newNodeList = matching(nodeList)
    print(len(newNodeList))
    level[levelId].append(newNodeList)
    levelId += 1

    while len(newNodeList) > 200:

        newNodeList = matching(newNodeList)
        print(len(newNodeList))
        level[levelId] = list()
        level[levelId] = newNodeList
        levelId += 1

    return level


def numRotar():

    trees = {}

    for num in range(9,10):

        print(num)
        particularNumTrainData = trainDataDict[num]
        trees[num] = bottomUpBuilder(particularNumTrainData=particularNumTrainData)

    with open('./pickles/treeDict.p','wb') as handle:
        pickle.dump(trees, handle)


# numRotar()