import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import gpiozero

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

backSubtractor = cv.createBackgroundSubtractorKNN()

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)


NUM_TRACKED_ENTRIES = 10
graphedContourCount = np.zeros(NUM_TRACKED_ENTRIES)
print(graphedContourCount)
oldestEntry = 0

recentMove = False
recentSpike = False
isWhiteTurn = True


def animate(i):
    # print(len(graphedContourCount))
    # print(list(range(0, len(graphedContourCount))))
    ax1.clear()
    reordered = []
    for i in range(NUM_TRACKED_ENTRIES):
        reordered.append(graphedContourCount[(oldestEntry + i) % NUM_TRACKED_ENTRIES])
    ax1.plot(list(range(len(graphedContourCount))), reordered)
    # ax1.plot([1, 2, 3], [1, 2, 3])


anim = ani.FuncAnimation(fig, animate, interval=250)

while True:
    # capture frame by frame
    ret, frame = cap.read()

    # if frame is read correctly ret is true
    if not ret:
        print("Can't receive frame. Exiting...")
        break

    fgMask = backSubtractor.apply(frame)
    fgMaskCopy = fgMask.copy()

    cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(frame, str(cap.get(cv.CAP_PROP_POS_FRAMES)), (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    # get contours from mask
    noErr, thresh = cv.threshold(fgMask, 254, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # print(contours[0])
    # print("------------------------------")

    significantContours = []
    threshold_area = 100
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > threshold_area:
            significantContours.append(cnt)

    if len(significantContours) > 12:
        print("-----LARGESPIKE-----", end="\n", flush=True)
        recentSpike = True

    graphedContourCount[oldestEntry] = (len(significantContours))
    oldestEntry = oldestEntry + 1
    oldestEntry = oldestEntry % NUM_TRACKED_ENTRIES

    avgRateOfChange = (graphedContourCount[oldestEntry-1] - graphedContourCount[oldestEntry]) / len(graphedContourCount)
    avgValue = np.average(graphedContourCount)
    totalDistance = 0
    for elem in graphedContourCount:
        totalDistance += abs(avgValue - elem)
    avgdistfromavg = totalDistance / NUM_TRACKED_ENTRIES
    # print("avgdistfromavg: " + str(avgdistfromavg) + " avgVal: " + str(avgValue), end="\n", flush=True)
    # print("avgrate: " + str(avgRateOfChange) + ", avgval: " + str(avgValue), end="\n", flush=True)

    for cnt in significantContours:
        averageX = 0
        averageY = 0
        for point in cnt:
            averageX += point[0][0]
            averageY += point[0][1]
        averageX /= len(cnt)
        averageY /= len(cnt)
        cv.circle(frame, (int(averageX), int(averageY)), 10, (0, 255, 0, 100), -1)
        M = cv.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    #     print(cx, cy)
    # print("---+++---")

    if avgRateOfChange < 0.1 and avgValue < 0.1 and (recentMove or recentSpike):
        print("****No activity, resetting move cooldown****", end="\n", flush=True)
        recentMove = False
        recentSpike = False

    if avgdistfromavg < 0.3 and avgValue >= 1 and not recentMove:
        print("-----\\\\\\\\MOVE//////-----", end="\n", flush=True)
        recentMove = True
        isWhiteTurn = not isWhiteTurn

    cv.imshow('frame', frame)
    cv.imshow('fgMask', fgMaskCopy)
    # print(thresh.shape)
    thresh = cv.drawContours(thresh, significantContours, -1, (255, 0, 0), 3)
    cv.imshow('contour', thresh)
    plt.show(block=False)
    plt.pause(0.01)
    if cv.waitKey(1) == ord('q'):
        break

plt.show()
cap.release()
cv.destroyAllWindows()
